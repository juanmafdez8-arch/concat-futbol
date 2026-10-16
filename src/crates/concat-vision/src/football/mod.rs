// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! Football footage: the players, keepers, referees and the ball, found
//! in every analysed instant of a media file and followed along it.
//!
//! The analysis is the host's job, in two passes. The first runs a YOLO
//! detector ([`detect`], behind `infer`) over the source at
//! [`crate::MASK_RATE`] frames a second and links what it finds into
//! tracks ([`track`]): one id per thing for as long as it stays in view.
//! That is a [`TrackSet`], kept as `tracks.json` in the media's football
//! folder in the project cache. The second pass is asked for per track,
//! when a mark needs the thing's silhouette rather than its box: the
//! brush model reads each instant the track is in and the masks go into a
//! [`MaskStore`] of their own beside it.
//!
//! The renderer reads both and never infers: [`marks`] draws a clip's
//! marks on its decoded frame, through the same [`crate::Mapping`] a
//! cutout uses, so a crop or a flip moves the marks with the picture.
//!
//! Every box is in source fractions, `[left, top, right, bottom]`, the
//! convention every mask in this crate keeps.

#[cfg(feature = "infer")]
pub mod detect;
pub mod draw;
pub mod marks;
pub mod track;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use crate::{MASK_RATE, MaskStore};

/// The id the ball always has: there is one ball, and a mark on it should
/// survive a re-analysis.
pub const BALL: u32 = 1;

/// What a detection is of.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// An outfield player.
    Player,
    /// A goalkeeper.
    Goalkeeper,
    /// A referee or assistant.
    Referee,
    /// The ball.
    Ball,
}

impl Kind {
    /// A person rather than the ball.
    pub fn is_person(self) -> bool {
        self != Kind::Ball
    }
}

/// One thing found in one frame.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Detection {
    /// What it is.
    pub kind: Kind,
    /// Where, as source fractions `[left, top, right, bottom]`.
    pub bbox: [f32; 4],
    /// How sure the detector was, `0..=1`.
    pub score: f32,
}

/// One tracked thing at one analysed instant.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub struct TrackedBox {
    /// The track's id; [`BALL`] for the ball.
    pub id: u32,
    /// What it is.
    #[serde(rename = "k")]
    pub kind: Kind,
    /// Where, as source fractions `[left, top, right, bottom]`.
    #[serde(rename = "b")]
    pub bbox: [f32; 4],
    /// The detector's score, or zero for a box filled in across a gap.
    #[serde(rename = "s")]
    pub score: f32,
}

impl TrackedBox {
    /// The box's centre, as source fractions.
    pub fn centre(&self) -> (f32, f32) {
        centre(&self.bbox)
    }
}

/// The analysed instants of one media file.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    /// The source instant, in milliseconds.
    pub ms: u64,
    /// What was there.
    pub boxes: Vec<TrackedBox>,
}

/// The football analysis of one media file: every analysed instant, in
/// order, with the tracked things in each.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TrackSet {
    /// The format's version.
    pub version: u32,
    /// The detector that found the boxes, for a reader that cares.
    #[serde(default)]
    pub model: String,
    /// The source's width over its height, for round marks.
    #[serde(default = "one")]
    pub aspect: f32,
    /// The analysed instants, ascending.
    pub samples: Vec<Sample>,
}

fn one() -> f32 {
    1.0
}

/// Further than this from an analysed instant, nothing is known.
const REACH_MS: u64 = 2500 / MASK_RATE as u64;

impl TrackSet {
    /// The boxes at `seconds` of source: each track that is in both the
    /// analysed instants around it, moved part of the way between them, so
    /// a mark glides at the footage's own rate rather than stepping at the
    /// analysis's. A track in only one of them is shown where it was, if
    /// that instant is near enough. A set of one instant is a still's - a
    /// frozen frame's - and holds at every time.
    pub fn at(&self, seconds: f64) -> Vec<TrackedBox> {
        if let [only] = self.samples.as_slice() {
            return only.boxes.clone();
        }
        let wanted = (seconds.max(0.0) * 1000.0).round() as u64;
        let index = self.samples.partition_point(|sample| sample.ms < wanted);
        let after = self.samples.get(index);
        let before = index.checked_sub(1).and_then(|i| self.samples.get(i));
        match (before, after) {
            (Some(a), Some(b)) if b.ms > a.ms => {
                let t = (wanted - a.ms) as f32 / (b.ms - a.ms) as f32;
                let near_a = wanted - a.ms <= REACH_MS;
                let near_b = b.ms - wanted <= REACH_MS;
                let mut out: Vec<TrackedBox> = Vec::new();
                for first in &a.boxes {
                    match b.boxes.iter().find(|other| other.id == first.id) {
                        Some(second) if b.ms - a.ms <= 4 * REACH_MS => out.push(TrackedBox {
                            bbox: lerp_box(&first.bbox, &second.bbox, t),
                            ..*first
                        }),
                        _ if near_a && t <= 0.5 => out.push(*first),
                        _ => {}
                    }
                }
                if near_b {
                    for second in &b.boxes {
                        if t > 0.5 && !out.iter().any(|held| held.id == second.id) {
                            out.push(*second);
                        }
                    }
                }
                out
            }
            (Some(only), _) | (_, Some(only)) if only.ms.abs_diff(wanted) <= REACH_MS => {
                only.boxes.clone()
            }
            _ => Vec::new(),
        }
    }

    /// Track `id` at `seconds`, if it is in view there.
    pub fn track_at(&self, id: u32, seconds: f64) -> Option<TrackedBox> {
        self.at(seconds).into_iter().find(|tracked| tracked.id == id)
    }

    /// The track under a point of the source at `seconds`, as fractions:
    /// the smallest box containing it, or failing that the nearest centre
    /// within a little of it - a click on a small, moving player should
    /// still find him.
    pub fn pick(&self, seconds: f64, u: f32, v: f32) -> Option<TrackedBox> {
        let boxes = self.at(seconds);
        let inside = boxes
            .iter()
            .filter(|tracked| {
                let [l, t, r, b] = grown(&tracked.bbox, 0.15);
                (l..=r).contains(&u) && (t..=b).contains(&v)
            })
            .min_by(|a, b| area(&a.bbox).total_cmp(&area(&b.bbox)));
        inside.copied().or_else(|| {
            boxes
                .iter()
                .map(|tracked| {
                    let (cx, cy) = tracked.centre();
                    (((cx - u) * self.aspect).hypot(cy - v), tracked)
                })
                .filter(|(distance, _)| *distance < 0.06)
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, tracked)| *tracked)
        })
    }

    /// Every track id with what it is and how many instants it is in,
    /// the ball first and the rest by how long they stay in view.
    pub fn tracks(&self) -> Vec<(u32, Kind, usize)> {
        let mut seen: HashMap<u32, (Kind, usize)> = HashMap::new();
        for sample in &self.samples {
            for tracked in &sample.boxes {
                seen.entry(tracked.id).or_insert((tracked.kind, 0)).1 += 1;
            }
        }
        let mut out: Vec<(u32, Kind, usize)> = seen
            .into_iter()
            .map(|(id, (kind, count))| (id, kind, count))
            .collect();
        out.sort_by(|a, b| {
            (b.0 == BALL)
                .cmp(&(a.0 == BALL))
                .then(b.2.cmp(&a.2))
                .then(a.0.cmp(&b.0))
        });
        out
    }

    /// The instants track `id` is in, with its box at each.
    pub fn instants_of(&self, id: u32) -> Vec<(u64, TrackedBox)> {
        self.samples
            .iter()
            .filter_map(|sample| {
                sample
                    .boxes
                    .iter()
                    .find(|tracked| tracked.id == id)
                    .map(|tracked| (sample.ms, *tracked))
            })
            .collect()
    }

    /// Reads `tracks.json` from a media's football folder; `None` when it
    /// is not analysed yet or the file cannot be read.
    pub fn load(dir: &Path) -> Option<TrackSet> {
        let bytes = std::fs::read(tracks_file(dir)).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    /// Writes the set into a media's football folder, by way of a file
    /// beside it so a reader never sees half of one.
    pub fn save(&self, dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dir)
            .map_err(|error| format!("could not create {}: {error}", dir.display()))?;
        let file = tracks_file(dir);
        let partial = file.with_extension("part");
        let bytes = serde_json::to_vec(self).map_err(|error| error.to_string())?;
        std::fs::write(&partial, bytes)
            .map_err(|error| format!("could not write {}: {error}", partial.display()))?;
        std::fs::rename(&partial, &file)
            .map_err(|error| format!("could not write {}: {error}", file.display()))?;
        forget(dir);
        Ok(())
    }
}

/// The folder a media file's football analysis lives in under `project`.
pub fn football_dir(project: &Path, media_path: &str) -> PathBuf {
    project
        .join("cache")
        .join("football")
        .join(format!("{:016x}", crate::store::fnv1a(media_path.as_bytes())))
}

/// The tracks file in a football folder.
pub fn tracks_file(dir: &Path) -> PathBuf {
    dir.join("tracks.json")
}

/// The folder track `id`'s silhouettes live in, inside a football folder.
pub fn silhouette_dir(dir: &Path, id: u32) -> PathBuf {
    dir.join("masks").join(id.to_string())
}

/// Track `id`'s silhouettes, as a mask store.
pub fn silhouettes(dir: &Path, id: u32) -> MaskStore {
    MaskStore::open(&silhouette_dir(dir, id))
}

/// Analyses read once per process and shared, keyed by folder and the
/// file's modification time so a fresh analysis is picked up.
fn cache() -> &'static Mutex<HashMap<PathBuf, (std::time::SystemTime, Arc<TrackSet>)>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, (std::time::SystemTime, Arc<TrackSet>)>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A media's analysis, from the process's cache or its file.
pub fn shared(dir: &Path) -> Option<Arc<TrackSet>> {
    let modified = std::fs::metadata(tracks_file(dir))
        .and_then(|meta| meta.modified())
        .ok()?;
    if let Ok(cache) = cache().lock()
        && let Some((when, set)) = cache.get(dir)
        && *when == modified
    {
        return Some(Arc::clone(set));
    }
    let set = Arc::new(TrackSet::load(dir)?);
    if let Ok(mut cache) = cache().lock() {
        cache.insert(dir.to_path_buf(), (modified, Arc::clone(&set)));
    }
    Some(set)
}

fn forget(dir: &Path) {
    if let Ok(mut cache) = cache().lock() {
        cache.remove(dir);
    }
}

/// A box's centre.
pub fn centre(bbox: &[f32; 4]) -> (f32, f32) {
    ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
}

/// A box's area.
pub fn area(bbox: &[f32; 4]) -> f32 {
    (bbox[2] - bbox[0]).max(0.0) * (bbox[3] - bbox[1]).max(0.0)
}

/// A box grown by `fraction` of its own size on every side.
pub fn grown(bbox: &[f32; 4], fraction: f32) -> [f32; 4] {
    let dx = (bbox[2] - bbox[0]) * fraction;
    let dy = (bbox[3] - bbox[1]) * fraction;
    [bbox[0] - dx, bbox[1] - dy, bbox[2] + dx, bbox[3] + dy]
}

/// Intersection over union of two boxes.
pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter = area(&[a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3])]);
    let union = area(a) + area(b) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Part of the way from one box to another.
pub fn lerp_box(a: &[f32; 4], b: &[f32; 4], t: f32) -> [f32; 4] {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set() -> TrackSet {
        TrackSet {
            version: 1,
            model: String::new(),
            aspect: 16.0 / 9.0,
            samples: vec![
                Sample {
                    ms: 0,
                    boxes: vec![TrackedBox {
                        id: 2,
                        kind: Kind::Player,
                        bbox: [0.1, 0.1, 0.2, 0.3],
                        score: 0.9,
                    }],
                },
                Sample {
                    ms: 100,
                    boxes: vec![TrackedBox {
                        id: 2,
                        kind: Kind::Player,
                        bbox: [0.2, 0.1, 0.3, 0.3],
                        score: 0.9,
                    }],
                },
            ],
        }
    }

    #[test]
    fn a_track_glides_between_instants() {
        let at = set().track_at(2, 0.05).expect("in view");
        assert!((at.bbox[0] - 0.15).abs() < 1e-5, "{:?}", at.bbox);
    }

    #[test]
    fn a_click_on_a_player_picks_him() {
        assert_eq!(set().pick(0.0, 0.15, 0.2).map(|t| t.id), Some(2));
        assert_eq!(set().pick(0.0, 0.8, 0.8), None);
    }

    #[test]
    fn a_still_holds_its_players_at_every_time() {
        let mut still = set();
        still.samples.truncate(1);
        assert_eq!(still.at(30.0).len(), 1);
    }

    #[test]
    fn nothing_is_known_far_from_the_analysis() {
        assert!(set().at(5.0).is_empty());
    }

    #[test]
    fn boxes_measure() {
        assert!((iou(&[0.0, 0.0, 1.0, 1.0], &[0.0, 0.0, 0.5, 1.0]) - 0.5).abs() < 1e-6);
        assert_eq!(iou(&[0.0, 0.0, 0.1, 0.1], &[0.5, 0.5, 0.6, 0.6]), 0.0);
    }
}

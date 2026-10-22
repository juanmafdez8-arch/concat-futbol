// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! A clip's football marks as the frame loop draws them.
//!
//! The analysis lives in the media's football folder in the project cache
//! (`concat_vision::football`); this reads it, finds each marked track at
//! the frame's source instant, and draws the marks on the decoded picture
//! before it is placed - the same frame the monitor and the export see,
//! like a cutout. An instant the analysis has not reached shows the
//! picture as shot.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use concat_core::frame::Frame;
use concat_core::time::Rational;
use concat_project::model::{DrawKind, Drawing, FootballMark, MarkStyle};
use concat_vision::football::{self, BALL, TrackSet, draw, marks};
use concat_vision::{Mapping, MaskStore};

use crate::ExportClip;

/// The colour every tracked box is outlined in while a player is picked,
/// the ball's and the chosen one's.
const OUTLINE: [u8; 3] = [0xff, 0xff, 0xff];
const OUTLINE_BALL: [u8; 3] = [0xff, 0x9f, 0x1c];
const OUTLINE_CHOSEN: [u8; 3] = [0xcb, 0xf5, 0x3f];

/// The order marks are drawn in: a hole is filled before anything is
/// drawn over it, and isolating comes last since it takes the rest away.
const ORDER: [MarkStyle; 7] = [
    MarkStyle::Erase,
    MarkStyle::Spotlight,
    MarkStyle::Shade,
    MarkStyle::Ring,
    MarkStyle::Frame,
    MarkStyle::Arrow,
    MarkStyle::Isolate,
];

/// A mark's shape knobs as the drawing functions take them.
fn shape_of(mark: &FootballMark) -> marks::Shape {
    marks::Shape {
        size: mark.size as f32,
        thickness: mark.thickness as f32,
        squash: mark.squash as f32,
        width: mark.width as f32,
    }
}

/// A mark's thing as the drawing functions take it.
fn target_of<'a>(
    tracked: &football::TrackedBox,
    silhouette: &'a Option<(std::sync::Arc<concat_vision::Mask>, (f32, f32))>,
) -> marks::Target<'a> {
    marks::Target {
        bbox: tracked.bbox,
        silhouette: silhouette.as_ref().map(|(mask, _)| mask.as_ref()),
        shift: silhouette
            .as_ref()
            .map(|(_, shift)| *shift)
            .unwrap_or((0.0, 0.0)),
    }
}

/// A clip's marks, ready to draw.
pub(crate) struct FootballJob {
    dir: PathBuf,
    marks: Vec<FootballMark>,
    drawings: Vec<Drawing>,
    /// Where the clip starts on the timeline: drawings are timed from it.
    start: f64,
    mapping: Mapping,
    picking: bool,
    chosen: Option<u32>,
    silhouettes: HashMap<u32, MaskStore>,
}

impl FootballJob {
    /// The job for a clip, or `None` when it has no marks and is not being
    /// picked from. `treated` as for a cutout: whether the frame has the
    /// clip's crop and flips in it already.
    pub(crate) fn of(clip: &ExportClip, treated: bool) -> Option<FootballJob> {
        let marked =
            !clip.football_dir.is_empty() && (!clip.football.is_empty() || clip.football_picking);
        if !marked && clip.drawings.is_empty() {
            return None;
        }
        let dir = PathBuf::from(&clip.football_dir);
        let silhouettes = clip
            .football
            .iter()
            .map(|mark| mark.track)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .map(|id| (id, football::silhouettes(&dir, id)))
            .collect();
        Some(FootballJob {
            dir,
            marks: clip.football.clone(),
            drawings: clip.drawings.clone(),
            start: clip.start,
            mapping: Mapping {
                crop: clip
                    .crop
                    .filter(|_| treated)
                    .map(|edges| edges.map(|edge| edge as f32))
                    .unwrap_or([0.0; 4]),
                flip_h: clip.flip_h && treated,
                flip_v: clip.flip_v && treated,
            },
            picking: clip.football_picking,
            chosen: clip.football_chosen,
            silhouettes,
        })
    }

    /// The frame with the marks and the drawings on it, or `None` when
    /// there is nothing to draw at this instant. `time` is the timeline's
    /// instant, which the drawings are timed by; `source_time` the
    /// footage's, which the players are found at.
    pub(crate) fn draw(
        &self,
        frame: &Frame,
        source_time: Rational,
        time: Rational,
    ) -> Option<Frame> {
        let set = if self.dir.as_os_str().is_empty() {
            None
        } else {
            football::shared(Path::new(&self.dir))
        };
        let seconds = source_time.as_f64();
        let sketches = self.sketches(set.as_deref(), seconds, time.as_f64() - self.start);
        let marked = set
            .as_deref()
            .and_then(|set| self.draw_marks(frame, set, seconds));
        if sketches.is_empty() {
            return marked;
        }
        let mut out = marked.unwrap_or_else(|| frame.clone());
        for sketch in &sketches {
            draw::draw(&mut out, sketch, &self.mapping);
        }
        Some(out)
    }

    /// The drawings showing `local` seconds into the clip, each point at
    /// the feet of the player it keeps to where he is in view.
    fn sketches(&self, set: Option<&TrackSet>, seconds: f64, local: f64) -> Vec<draw::Sketch> {
        /// How long a drawing takes to fade at the end of its hold.
        const FADE: f64 = 0.25;
        self.drawings
            .iter()
            .filter(|drawing| drawing.start <= local && local < drawing.start + drawing.hold)
            .map(|drawing| {
                let shown = local - drawing.start;
                let reveal = if drawing.reveal > 0.0 {
                    (shown / drawing.reveal).min(1.0)
                } else {
                    1.0
                };
                let left = drawing.start + drawing.hold - local;
                let points = drawing
                    .points
                    .iter()
                    .map(|anchor| {
                        anchor
                            .track
                            .and_then(|track| set?.track_at(track, seconds))
                            .map(|tracked| {
                                let [l, _, r, b] = tracked.bbox;
                                ((l + r) * 0.5, b)
                            })
                            .unwrap_or((anchor.at[0] as f32, anchor.at[1] as f32))
                    })
                    .collect();
                draw::Sketch {
                    kind: match drawing.kind {
                        DrawKind::Line => draw::Kind::Line,
                        DrawKind::Arrow => draw::Kind::Arrow,
                        DrawKind::Curve => draw::Kind::Curve,
                        DrawKind::Zone => draw::Kind::Zone,
                        DrawKind::Link => draw::Kind::Link,
                    },
                    points,
                    dashed: drawing.dashed,
                    colour: drawing.colour,
                    thickness: drawing.thickness as f32,
                    bend: drawing.bend as f32,
                    reveal: reveal as f32,
                    opacity: (left / FADE).min(1.0) as f32,
                }
            })
            .collect()
    }

    /// The frame with the football marks drawn, or `None` when there is
    /// nothing to draw at this instant.
    fn draw_marks(&self, frame: &Frame, set: &TrackSet, seconds: f64) -> Option<Frame> {
        if self.marks.is_empty() && !self.picking {
            return None;
        }
        let boxes = set.at(seconds);
        let boxes_by_id: HashMap<_, _> =
            boxes.iter().map(|tracked| (tracked.id, *tracked)).collect();
        let isolating = self
            .marks
            .iter()
            .any(|mark| mark.style == MarkStyle::Isolate);
        if boxes.is_empty() && !isolating {
            return None;
        }
        let mut out = frame.clone();
        // Each mark's thing at this instant, with its silhouette: read once,
        // since a spotlight wants them all at once.
        let mut silhouettes = HashMap::new();
        let found: Vec<_> = self
            .marks
            .iter()
            .map(|mark| {
                let tracked = boxes_by_id.get(&mark.track).copied();
                let silhouette = tracked.and_then(|tracked| {
                    silhouettes
                        .entry(mark.track)
                        .or_insert_with(|| {
                            self.silhouette(set, mark.track, seconds, tracked.centre())
                        })
                        .clone()
                });
                (mark, tracked, silhouette)
            })
            .collect();
        for style in ORDER {
            let of_style = found.iter().filter(|(mark, _, _)| mark.style == style);
            if style == MarkStyle::Spotlight {
                let targets: Vec<(marks::Target<'_>, marks::Shape)> = of_style
                    .clone()
                    .filter_map(|(mark, tracked, silhouette)| {
                        tracked
                            .as_ref()
                            .map(|tracked| (target_of(tracked, silhouette), shape_of(mark)))
                    })
                    .collect();
                let strength = of_style
                    .map(|(mark, _, _)| mark.strength as f32)
                    .fold(0.0, f32::max);
                marks::spotlight(&mut out, &targets, &self.mapping, set.aspect, strength);
                continue;
            }
            for (mark, tracked, silhouette) in of_style {
                let Some(tracked) = tracked else {
                    if style == MarkStyle::Isolate {
                        // The player has left the picture: so has the layer.
                        for pixel in out.pixels_mut().chunks_exact_mut(4) {
                            pixel[3] = 0;
                        }
                    }
                    continue;
                };
                let target = target_of(tracked, silhouette);
                let strength = mark.strength as f32;
                match style {
                    MarkStyle::Erase => {
                        marks::erase(&mut out, &target, &self.mapping, mark.size as f32);
                    }
                    MarkStyle::Shade => {
                        marks::shade(&mut out, &target, &self.mapping, mark.colour, strength);
                    }
                    MarkStyle::Ring => marks::ring(
                        &mut out,
                        &target,
                        &self.mapping,
                        set.aspect,
                        mark.colour,
                        strength,
                        shape_of(mark),
                    ),
                    MarkStyle::Arrow => marks::arrow(
                        &mut out,
                        &target,
                        &self.mapping,
                        set.aspect,
                        mark.colour,
                        strength,
                        shape_of(mark),
                    ),
                    MarkStyle::Frame => marks::frame_round(
                        &mut out,
                        &target,
                        &self.mapping,
                        mark.colour,
                        strength,
                        shape_of(mark),
                    ),
                    MarkStyle::Isolate => marks::isolate(&mut out, &target, &self.mapping),
                    MarkStyle::Spotlight => {}
                }
            }
        }
        if self.picking {
            let outlined: Vec<([f32; 4], [u8; 3], bool)> = boxes
                .iter()
                .map(|tracked| {
                    let chosen = Some(tracked.id) == self.chosen;
                    let colour = if chosen {
                        OUTLINE_CHOSEN
                    } else if tracked.id == BALL {
                        OUTLINE_BALL
                    } else {
                        OUTLINE
                    };
                    (tracked.bbox, colour, chosen)
                })
                .collect();
            marks::outline(&mut out, &outlined, &self.mapping);
        }
        Some(out)
    }

    /// Track `id`'s silhouette nearest `seconds`, with how far the track
    /// has moved since that silhouette's instant.
    fn silhouette(
        &self,
        set: &TrackSet,
        id: u32,
        seconds: f64,
        now: (f32, f32),
    ) -> Option<(std::sync::Arc<concat_vision::Mask>, (f32, f32))> {
        let store = self.silhouettes.get(&id)?;
        let millis = store.nearest_millis(seconds)?;
        let mask = store.mask_at(seconds)?;
        let then = set.track_at(id, millis as f64 / 1000.0)?.centre();
        Some((mask, (now.0 - then.0, now.1 - then.1)))
    }
}

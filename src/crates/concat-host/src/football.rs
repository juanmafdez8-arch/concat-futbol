// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! Finding the players and the ball in football footage, and reading a
//! player's silhouette.
//!
//! Two jobs, one slot. [`Footballs::analyse`] decodes the stretch of a
//! media file its clips show at [`MASK_RATE`] frames a second, runs the
//! detectors on each frame, links what they find into tracks and writes
//! the result as the media's `tracks.json`. [`Footballs::silhouettes`] is
//! asked for one track once a mark wants its outline: it decodes the
//! instants the track is in at the source's own size, has the brush model
//! read a crop round the player in each, and writes the masks beside the
//! tracks.
//!
//! The detectors are YOLO models exported to ONNX at [`DETECT_SIZE`]
//! pixels square, kept in `<app data>/football-models/`: `football.onnx`
//! for the people (a model trained on football, with goalkeepers and
//! referees told apart) and, if it is there, `general.onnx` for the ball,
//! which a general model finds more often on broadcast footage. Each may
//! have a `.classes.json` beside it naming its classes in order.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use concat_core::time::{FrameRate, Rational};
use concat_media::{DecodeOptions, Decoder, FrameSource};
use concat_project::model::Project;
use concat_vision::MASK_RATE;
use concat_vision::football::detect::{Detectors, silhouette};
use concat_vision::football::{TrackSet, football_dir, silhouette_dir, track, tracks_file};

use crate::brush::Brushes;
use crate::cutout::Progress;
use crate::jobs::SingleFlight;

/// The square the detectors were exported at.
pub const DETECT_SIZE: usize = 1280;
/// The width a silhouette is stored at, over the whole source.
const SILHOUETTE_WIDTH: u32 = 1280;
/// The widest a frame is decoded at for the brush model to crop from.
const READ_WIDTH: u32 = 1920;

/// What one analysis covers.
#[derive(Clone, Debug)]
pub struct FootballRequest {
    /// The project folder the analysis is cached under.
    pub project: PathBuf,
    /// The media file to analyse.
    pub media_path: String,
    /// The media's picture size.
    pub media_size: (u32, u32),
    /// The stretches of source, in seconds, that clips show.
    pub ranges: Vec<(f64, f64)>,
}

impl FootballRequest {
    /// The media's football folder.
    pub fn dir(&self) -> PathBuf {
        football_dir(&self.project, &self.media_path)
    }

    /// Whether the media has been analysed.
    pub fn analysed(&self) -> bool {
        tracks_file(&self.dir()).is_file()
    }

    /// The analysis for the clip `clip_id` of `project`, if it shows a
    /// picture from a file.
    pub fn for_clip(project: &Project, project_dir: &Path, clip_id: &str) -> Option<FootballRequest> {
        let clip = project.active().clips.iter().find(|clip| clip.id == clip_id)?;
        if !clip.kind.is_visual() {
            return None;
        }
        let media = project.media_by_id(&clip.media_id)?;
        // Every clip of the same file, so one analysis serves them all.
        let ranges = project
            .active()
            .clips
            .iter()
            .filter(|other| other.media_id == media.id)
            .map(|other| {
                (
                    other.source_start,
                    other.source_start + other.duration * other.speed.max(0.0625),
                )
            })
            .collect();
        Some(FootballRequest {
            project: project_dir.to_path_buf(),
            media_path: media.path.clone(),
            media_size: (media.width.unwrap_or(0), media.height.unwrap_or(0)),
            ranges,
        })
    }
}

/// The football service: the detectors, loaded once, and the one-job slot.
pub struct Footballs {
    gate: Arc<SingleFlight>,
    data: PathBuf,
    detectors: Mutex<Option<Arc<Detectors>>>,
    brushes: Arc<Brushes>,
}

impl Footballs {
    /// A service with nothing loaded, finding its models under `data` and
    /// borrowing the brush model from `brushes`.
    pub fn new(data: &Path, brushes: Arc<Brushes>) -> Footballs {
        Footballs {
            gate: Arc::new(SingleFlight::new()),
            data: data.to_path_buf(),
            detectors: Mutex::new(None),
            brushes,
        }
    }

    /// Where the detectors live.
    pub fn models_dir(data: &Path) -> PathBuf {
        data.join("football-models")
    }

    /// Whether the football detector is on disk.
    pub fn installed(&self) -> bool {
        Self::models_dir(&self.data).join("football.onnx").is_file()
    }

    /// Whether a job is running.
    pub fn is_busy(&self) -> bool {
        self.gate.is_busy()
    }

    /// Asks the running job to stop.
    pub fn cancel(&self) {
        self.gate.cancel();
    }

    fn detectors(&self) -> Result<Arc<Detectors>, String> {
        let mut slot = self
            .detectors
            .lock()
            .map_err(|_| "football detectors poisoned")?;
        if let Some(loaded) = slot.as_ref() {
            return Ok(Arc::clone(loaded));
        }
        let dir = Self::models_dir(&self.data);
        let people = dir.join("football.onnx");
        if !people.is_file() {
            return Err(format!(
                "no football model: put football.onnx (YOLO exported to ONNX at {DETECT_SIZE}) in {}",
                dir.display()
            ));
        }
        let general = dir.join("general.onnx");
        let general = general.is_file().then_some(general);
        let loaded = Arc::new(Detectors::load(&people, general.as_deref(), DETECT_SIZE)?);
        *slot = Some(Arc::clone(&loaded));
        Ok(loaded)
    }

    /// Finds and tracks the players and the ball over the stretch of
    /// source the request's clips show, and writes the result. Blocks for
    /// the whole run; `progress` is called as it goes. Returns how many
    /// instants were analysed.
    pub fn analyse(
        &self,
        request: &FootballRequest,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<usize, String> {
        let job = self.gate.begin("football analysis")?;
        let detectors = self.detectors()?;
        let from = request
            .ranges
            .iter()
            .map(|range| range.0)
            .fold(f64::MAX, f64::min)
            .max(0.0);
        let to = request
            .ranges
            .iter()
            .map(|range| range.1)
            .fold(0.0, f64::max);
        if to <= from {
            return Ok(0);
        }
        let step_ms = 1000 / u64::from(MASK_RATE);
        let first_ms = ((from * 1000.0) as u64) / step_ms * step_ms;
        let count = (((to * 1000.0) as u64).saturating_sub(first_ms) / step_ms + 1) as usize;
        let (width, height) = request.media_size;
        let (w, h) = detectors.input_size(width.max(1), height.max(1));
        let options = DecodeOptions::default()
            .starting_at(Rational::new(first_ms as i64, 1000))
            .scaled_to(w, h)
            .at_rate(FrameRate::new(Rational::new(i64::from(MASK_RATE), 1)))
            .limited_to(count as u64);
        let mut decoder =
            Decoder::open(&request.media_path, &options).map_err(|error| error.to_string())?;
        progress(Progress::Analysing(0.0));
        let mut frames = Vec::with_capacity(count);
        for index in 0..count {
            if job.cancelled() {
                return Err("football analysis cancelled".to_owned());
            }
            let Some(frame) = decoder.next_frame().map_err(|error| error.to_string())? else {
                break;
            };
            let found = detectors.detect(&frame)?;
            frames.push((first_ms + index as u64 * step_ms, found));
            // The last tenth is the linking and the writing.
            progress(Progress::Analysing(0.9 * (index + 1) as f32 / count as f32));
        }
        let analysed = frames.len();
        let dir = request.dir();
        let mut samples = track::link(&frames, step_ms);
        // The players keep the ids the last analysis gave them, so the
        // marks clips carry still name the same people.
        if let Some(before) = TrackSet::load(&dir) {
            track::keep_ids(&before.samples, &mut samples);
        }
        let set = TrackSet {
            version: 1,
            model: detectors.name(),
            aspect: if height > 0 {
                width as f32 / height as f32
            } else {
                w as f32 / h.max(1) as f32
            },
            samples,
        };
        set.save(&dir)?;
        // Only once the new analysis is safely written: the old silhouettes
        // were read inside the old boxes, and are read again on demand.
        let _ = std::fs::remove_dir_all(dir.join("masks"));
        progress(Progress::Analysing(1.0));
        Ok(analysed)
    }

    /// Reads track `id`'s silhouette at every instant it is in, and writes
    /// them. Blocks; `progress` as for [`Footballs::analyse`]. Returns how
    /// many were written.
    pub fn silhouettes(
        &self,
        request: &FootballRequest,
        id: u32,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<usize, String> {
        let job = self.gate.begin("football silhouettes")?;
        let cancel = job.cancel_handle();
        let dir = request.dir();
        let set = TrackSet::load(&dir).ok_or("the footage has not been analysed yet")?;
        let instants = set.instants_of(id);
        if instants.is_empty() {
            return Ok(0);
        }
        let brush = self.brushes.brush(&cancel, progress)?;
        let mut store = concat_vision::MaskStore::open(&silhouette_dir(&dir, id));
        let wanted: Vec<_> = instants
            .into_iter()
            .filter(|(ms, _)| store.nearest_millis(*ms as f64 / 1000.0) != Some(*ms))
            .collect();
        if wanted.is_empty() {
            return Ok(0);
        }
        let step_ms = 1000 / u64::from(MASK_RATE);
        let first_ms = wanted[0].0;
        let last_ms = wanted[wanted.len() - 1].0;
        let count = ((last_ms - first_ms) / step_ms + 1) as u64;
        let (width, height) = request.media_size;
        let (w, h) = if width > READ_WIDTH && width > 0 {
            (READ_WIDTH, ((height as f64 * READ_WIDTH as f64 / width as f64).round() as u32).max(1))
        } else {
            (width.max(1), height.max(1))
        };
        let options = DecodeOptions::default()
            .starting_at(Rational::new(first_ms as i64, 1000))
            .scaled_to(w, h)
            .at_rate(FrameRate::new(Rational::new(i64::from(MASK_RATE), 1)))
            .limited_to(count);
        let mut decoder =
            Decoder::open(&request.media_path, &options).map_err(|error| error.to_string())?;
        if store.model().is_none() {
            store.set_model("slimsam-football")?;
        }
        progress(Progress::Analysing(0.0));
        let mut next = 0usize;
        let mut written = 0usize;
        let mut at_ms = first_ms;
        while next < wanted.len() {
            if job.cancelled() {
                return Err("football silhouettes cancelled".to_owned());
            }
            let Some(frame) = decoder.next_frame().map_err(|error| error.to_string())? else {
                break;
            };
            if at_ms == wanted[next].0 {
                let tracked = wanted[next].1;
                // A silhouette the model cannot read is left out: the mark
                // falls back to the box there.
                if let Ok(mask) = silhouette(&brush, &frame, tracked.bbox, SILHOUETTE_WIDTH) {
                    store.put(at_ms, &mask)?;
                    written += 1;
                }
                next += 1;
                progress(Progress::Analysing(next as f32 / wanted.len() as f32));
            }
            at_ms += step_ms;
            // Skip past instants the track is not in.
            while next < wanted.len() && wanted[next].0 < at_ms {
                next += 1;
            }
        }
        progress(Progress::Analysing(1.0));
        Ok(written)
    }
}

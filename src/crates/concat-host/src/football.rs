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
use concat_vision::football::detect::{Detectors, silhouettes};
use concat_vision::football::{TrackSet, football_dir, pitch, silhouette_dir, track, tracks_file};

use crate::brush::Brushes;
use crate::cutout::Progress;
use crate::jobs::SingleFlight;

/// The square the detectors were exported at.
pub const DETECT_SIZE: usize = 1280;
/// The width a silhouette is stored at, over the whole source.
const SILHOUETTE_WIDTH: u32 = 1280;
/// The widest a frame is decoded at for the brush model to crop from.
const READ_WIDTH: u32 = 1920;
/// The analysis's step: five instants a second. Measured on broadcast
/// footage, the tracker keeps the same identities as at ten, and the
/// renderer glides each mark between the instants, so the other half of
/// the detector's work bought nothing.
const STEP_MS: u64 = 200;
/// The window analysed first round the instant being looked at, before
/// and after it, in seconds.
const WINDOW_BEFORE: f64 = 4.0;
const WINDOW_AFTER: f64 = 16.0;

/// One analysed instant's detections.
type Detections = (u64, Vec<concat_vision::football::Detection>);
/// Each outfield player's box and shirt colour at one instant.
type Seen = Vec<([f32; 4], [u8; 3])>;
/// One analysed instant's shirts.
type Shirts = (u64, Seen);
/// A stretch's detections and shirts.
type Spans = (Vec<Detections>, Vec<Shirts>);

/// The players of `samples` told apart into two teams by their shirts,
/// each box labelled; the two kits' colours, or none when there were too
/// few players to tell.
fn label_teams(samples: &mut [concat_vision::football::Sample], shirts: &[Shirts]) -> Vec<[u8; 3]> {
    let by_ms: std::collections::HashMap<u64, &Seen> =
        shirts.iter().map(|(ms, seen)| (*ms, seen)).collect();
    let mut colours: std::collections::HashMap<u32, Vec<[u8; 3]>> = std::collections::HashMap::new();
    for sample in samples.iter() {
        let Some(seen) = by_ms.get(&sample.ms) else {
            continue;
        };
        for tracked in sample.boxes.iter().filter(|t| pitch::in_teams(t.kind) && t.score > 0.0) {
            // The tracker keeps a matched detection's box as it was.
            if let Some((_, colour)) = seen.iter().find(|(bbox, _)| *bbox == tracked.bbox) {
                colours.entry(tracked.id).or_default().push(*colour);
            }
        }
    }
    let (teams, kits) = pitch::teams(&colours);
    if teams.is_empty() {
        return Vec::new();
    }
    for sample in samples.iter_mut() {
        for tracked in &mut sample.boxes {
            tracked.team = teams.get(&tracked.id).copied().unwrap_or(0);
        }
    }
    kits.to_vec()
}

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
    ///
    /// With a `focus` - the source instant the person is looking at - a
    /// window round it is analysed and written first, and `on_window` is
    /// called, so a click waiting on the players lands in a second or two
    /// rather than after the whole match; then the rest, which keeps the
    /// window's detections and the ids it gave.
    pub fn analyse(
        &self,
        request: &FootballRequest,
        focus: Option<f64>,
        progress: &mut dyn FnMut(Progress),
        on_window: &mut dyn FnMut(),
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
        let dir = request.dir();
        let before = TrackSet::load(&dir);
        let (width, height) = request.media_size;
        let (w, h) = detectors.input_size(width.max(1), height.max(1));
        let aspect = if height > 0 {
            width as f32 / height as f32
        } else {
            w as f32 / h.max(1) as f32
        };
        let save = |samples: Vec<concat_vision::football::Sample>, team_colours: Vec<[u8; 3]>| {
            TrackSet {
                version: 1,
                model: detectors.name(),
                aspect,
                samples,
                team_colours,
            }
            .save(&dir)
        };

        // The window round the focus, when the stretch is long enough for
        // it to matter: written as soon as it is found.
        let window = focus
            .filter(|_| to - from > WINDOW_BEFORE + WINDOW_AFTER + 10.0)
            .map(|focus| {
                (
                    (focus - WINDOW_BEFORE).max(from),
                    (focus + WINDOW_AFTER).min(to),
                )
            });
        let total = to - from;
        let mut done = 0.0;
        let mut frames: Vec<(u64, Vec<concat_vision::football::Detection>)> = Vec::new();
        let mut shirts: Vec<Shirts> = Vec::new();
        let mut first_ids: Option<Vec<concat_vision::football::Sample>> = None;
        if let Some((window_from, window_to)) = window {
            let span = window_to - window_from;
            (frames, shirts) = self.detect_span(&job, &detectors, request, (w, h), window_from, window_to, &mut |f| {
                progress(Progress::Analysing((0.9 * f as f64 * span / total) as f32));
            })?;
            done += span;
            let mut samples = track::link(&frames, STEP_MS);
            if let Some(before) = &before {
                track::keep_ids(&before.samples, &mut samples);
            }
            let kits = label_teams(&mut samples, &shirts);
            save(samples.clone(), kits)?;
            first_ids = Some(samples);
            on_window();
        }

        // The rest: before the window and after it, or all of it.
        let rest: Vec<(f64, f64)> = match window {
            Some((window_from, window_to)) => [(from, window_from), (window_to, to)]
                .into_iter()
                .filter(|(a, b)| b - a > 0.05)
                .collect(),
            None => vec![(from, to)],
        };
        for (span_from, span_to) in rest {
            let span = span_to - span_from;
            let base = done;
            let (mut found, mut seen) = self.detect_span(&job, &detectors, request, (w, h), span_from, span_to, &mut |f| {
                progress(Progress::Analysing((0.9 * (base + f as f64 * span) / total) as f32));
            })?;
            done += span;
            frames.append(&mut found);
            shirts.append(&mut seen);
        }
        frames.sort_by_key(|(ms, _)| *ms);
        frames.dedup_by_key(|(ms, _)| *ms);
        let analysed = frames.len();
        let mut samples = track::link(&frames, STEP_MS);
        // The players keep the ids the window - or the last analysis -
        // gave them, so the marks clips carry still name the same people.
        if let Some(earlier) = first_ids.as_ref().or(before.as_ref().map(|set| &set.samples)) {
            track::keep_ids(earlier, &mut samples);
        }
        let kits = label_teams(&mut samples, &shirts);
        save(samples, kits)?;
        // Only once the new analysis is safely written: the old silhouettes
        // were read inside the old boxes, and are read again on demand.
        if before.is_some() {
            let _ = std::fs::remove_dir_all(dir.join("masks"));
        }
        progress(Progress::Analysing(1.0));
        Ok(analysed)
    }

    /// The detections in `[from, to]` seconds of source at the analysis
    /// rate. The decoder runs a few frames ahead on a thread of its own, so
    /// decoding and the detectors overlap instead of taking turns.
    fn detect_span(
        &self,
        job: &crate::jobs::Job,
        detectors: &Detectors,
        request: &FootballRequest,
        (w, h): (u32, u32),
        from: f64,
        to: f64,
        progress: &mut dyn FnMut(f32),
    ) -> Result<Spans, String> {
        let first_ms = ((from.max(0.0) * 1000.0) as u64).div_ceil(STEP_MS) * STEP_MS;
        let last_ms = (to.max(0.0) * 1000.0) as u64;
        if last_ms < first_ms {
            return Ok((Vec::new(), Vec::new()));
        }
        let count = ((last_ms - first_ms) / STEP_MS + 1) as usize;
        let options = DecodeOptions::default()
            .starting_at(Rational::new(first_ms as i64, 1000))
            .scaled_to(w, h)
            .at_rate(FrameRate::new(Rational::new((1000 / STEP_MS) as i64, 1)))
            .limited_to(count as u64);
        let path = request.media_path.clone();
        let (sender, receiver) = std::sync::mpsc::sync_channel::<Result<concat_core::frame::Frame, String>>(4);
        let mut out = Vec::with_capacity(count);
        let mut shirts = Vec::with_capacity(count);
        let mut failure: Option<String> = None;
        std::thread::scope(|scope| {
            scope.spawn(move || {
                let mut decoder = match Decoder::open(&path, &options) {
                    Ok(decoder) => decoder,
                    Err(error) => {
                        let _ = sender.send(Err(error.to_string()));
                        return;
                    }
                };
                loop {
                    match decoder.next_frame() {
                        Ok(Some(frame)) => {
                            if sender.send(Ok(frame)).is_err() {
                                return;
                            }
                        }
                        Ok(None) => return,
                        Err(error) => {
                            let _ = sender.send(Err(error.to_string()));
                            return;
                        }
                    }
                }
            });
            for index in 0..count {
                if job.cancelled() {
                    failure = Some("football analysis cancelled".to_owned());
                    break;
                }
                let frame = match receiver.recv() {
                    Ok(Ok(frame)) => frame,
                    Ok(Err(error)) => {
                        failure = Some(error);
                        break;
                    }
                    // The file ran out before its stated length.
                    Err(_) => break,
                };
                match detectors.detect(&frame) {
                    Ok(found) => {
                        let ms = first_ms + index as u64 * STEP_MS;
                        // The shirts while the frame is here: what the
                        // teams are told apart by after the linking.
                        let seen: Vec<([f32; 4], [u8; 3])> = found
                            .iter()
                            .filter(|detection| pitch::in_teams(detection.kind))
                            .filter_map(|detection| {
                                pitch::shirt_colour(&frame, &detection.bbox)
                                    .map(|colour| (detection.bbox, colour))
                            })
                            .collect();
                        shirts.push((ms, seen));
                        out.push((ms, found));
                    }
                    Err(error) => {
                        failure = Some(error);
                        break;
                    }
                }
                progress((index + 1) as f32 / count as f32);
            }
            // Dropping the receiver ends the decoder's thread at its next
            // send, so the scope is never left waiting on a full channel.
            drop(receiver);
        });
        match failure {
            Some(error) => Err(error),
            None => Ok((out, shirts)),
        }
    }

    /// Reads track `id`'s silhouette at every instant it is in, and writes
    /// them. Blocks; `progress` as for [`Footballs::analyse`]. Returns how
    /// many were written.
    ///
    /// Four instants go to the brush model at a time, laid out on one
    /// square (see [`silhouettes`]), which is where nearly all the time
    /// goes; the renderer moves each silhouette with its player between
    /// the instants.
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
        let mut store = concat_vision::MaskStore::open(&silhouette_dir(&dir, id));
        let wanted: Vec<_> = set
            .instants_of(id)
            .into_iter()
            .filter(|(ms, _)| ms % STEP_MS == 0)
            .filter(|(ms, _)| store.nearest_millis(*ms as f64 / 1000.0) != Some(*ms))
            .collect();
        if wanted.is_empty() {
            return Ok(0);
        }
        let brush = self.brushes.brush(&cancel, progress)?;
        let first_ms = wanted[0].0;
        let last_ms = wanted[wanted.len() - 1].0;
        let count = (last_ms - first_ms) / STEP_MS + 1;
        let (width, height) = request.media_size;
        let (w, h) = if width > READ_WIDTH {
            (READ_WIDTH, ((height as f64 * READ_WIDTH as f64 / width as f64).round() as u32).max(1))
        } else {
            (width.max(1), height.max(1))
        };
        let options = DecodeOptions::default()
            .starting_at(Rational::new(first_ms as i64, 1000))
            .scaled_to(w, h)
            .at_rate(FrameRate::new(Rational::new((1000 / STEP_MS) as i64, 1)))
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
        let mut batch: Vec<(u64, concat_core::frame::Frame, [f32; 4])> = Vec::new();
        let flush = |batch: &mut Vec<(u64, concat_core::frame::Frame, [f32; 4])>,
                         store: &mut concat_vision::MaskStore|
         -> Result<usize, String> {
            if batch.is_empty() {
                return Ok(0);
            }
            let shots: Vec<(&concat_core::frame::Frame, [f32; 4])> =
                batch.iter().map(|(_, frame, bbox)| (frame, *bbox)).collect();
            // A batch the model cannot read is left out: the marks fall
            // back to the box there.
            let masks = silhouettes(&brush, &shots, SILHOUETTE_WIDTH).unwrap_or_default();
            let mut put = 0;
            for ((ms, _, _), mask) in batch.iter().zip(masks) {
                if let Some(mask) = mask {
                    store.put(*ms, &mask)?;
                    put += 1;
                }
            }
            batch.clear();
            Ok(put)
        };
        while next < wanted.len() {
            if job.cancelled() {
                return Err("football silhouettes cancelled".to_owned());
            }
            let Some(frame) = decoder.next_frame().map_err(|error| error.to_string())? else {
                break;
            };
            if at_ms == wanted[next].0 {
                batch.push((at_ms, frame, wanted[next].1.bbox));
                next += 1;
                if batch.len() == 4 {
                    written += flush(&mut batch, &mut store)?;
                    progress(Progress::Analysing(next as f32 / wanted.len() as f32));
                }
            }
            at_ms += STEP_MS;
            while next < wanted.len() && wanted[next].0 < at_ms {
                next += 1;
            }
        }
        written += flush(&mut batch, &mut store)?;
        progress(Progress::Analysing(1.0));
        Ok(written)
    }
}

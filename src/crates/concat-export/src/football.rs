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
use std::sync::{Arc, Mutex, OnceLock};

use concat_core::frame::Frame;
use concat_core::time::Rational;
use concat_project::model::{DrawKind, Drawing, FootballMark, MarkStyle};
use concat_vision::football::{self, BALL, TrackSet, draw, marks, pitch};
use concat_vision::{Mapping, MaskStore};

use crate::ExportClip;

/// The colour every tracked box is outlined in while a player is picked,
/// the ball's and the chosen one's.
const OUTLINE: [u8; 3] = [0xff, 0xff, 0xff];
const OUTLINE_BALL: [u8; 3] = [0xff, 0x9f, 0x1c];
const OUTLINE_CHOSEN: [u8; 3] = [0xcb, 0xf5, 0x3f];

/// The order marks are drawn in: a hole is filled before anything is
/// drawn over it, and isolating comes last since it takes the rest away.
/// The pitch's own lines go down first, on the grass under everything; the
/// labels go last but one, over the marks they name.
const ORDER: [MarkStyle; 13] = [
    MarkStyle::Erase,
    MarkStyle::Lanes,
    MarkStyle::LastLine,
    MarkStyle::Spotlight,
    MarkStyle::Shade,
    MarkStyle::Trail,
    MarkStyle::Distance,
    MarkStyle::Ring,
    MarkStyle::Frame,
    MarkStyle::Arrow,
    MarkStyle::Lens,
    MarkStyle::Speed,
    MarkStyle::Isolate,
];

/// How far back a trail reaches at size 1, in milliseconds.
const TRAIL_MS: f64 = 2500.0;
/// The pitch, in metres.
const PITCH: (f64, f64) = (105.0, 68.0);
/// A distance to the ball reads green this near, yellow up to `FAR`, and
/// red beyond: the press the model asks for is within ten metres.
const NEAR: f64 = 5.0;
const FAR: f64 = 10.0;

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
    /// The clip's calibration, for the marks that read metres.
    calibration: pitch::Calibration,
    /// The camera's motion through the analysis, measured once.
    motion: OnceLock<Arc<pitch::Motion>>,
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
            calibration: pitch::Calibration::load(&dir),
            motion: OnceLock::new(),
            dir,
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
                    MarkStyle::Lens => marks::lens(
                        &mut out,
                        &target,
                        &self.mapping,
                        mark.colour,
                        strength,
                        shape_of(mark),
                    ),
                    MarkStyle::Trail => self.trail(&mut out, set, mark, tracked, seconds),
                    MarkStyle::Distance => {
                        self.distance(&mut out, set, mark, tracked, &boxes_by_id, seconds)
                    }
                    MarkStyle::Speed => self.speed(&mut out, set, mark, tracked, seconds),
                    MarkStyle::LastLine => self.last_line(&mut out, set, mark, tracked, seconds),
                    MarkStyle::Lanes => self.lanes(&mut out, set, mark, seconds),
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


    /// The camera's motion through `set`, measured the first time a mark
    /// asks for it.
    fn motion(&self, set: &TrackSet) -> Arc<pitch::Motion> {
        Arc::clone(self.motion.get_or_init(|| Arc::new(pitch::Motion::of(set))))
    }

    /// The picture to pitch metres at `ms`, when the clip is calibrated.
    fn to_pitch(&self, set: &TrackSet, ms: u64) -> Option<pitch::Matrix> {
        pitch::map_at(&self.calibration, &self.motion(set), ms)
    }

    /// Pitch metres to the picture at `ms`, when the clip is calibrated.
    fn to_picture(&self, set: &TrackSet, ms: u64) -> Option<pitch::Matrix> {
        pitch::image_map_at(&self.calibration, &self.motion(set), ms)
    }

    /// A line through `points`, in source fractions.
    fn line(
        &self,
        out: &mut Frame,
        points: Vec<(f32, f32)>,
        kind: draw::Kind,
        colour: [u8; 3],
        thickness: f32,
        dashed: bool,
        opacity: f32,
    ) {
        draw::draw(
            out,
            &draw::Sketch {
                kind,
                points,
                dashed,
                colour,
                thickness,
                bend: 0.0,
                reveal: 1.0,
                opacity,
            },
            &self.mapping,
        );
    }

    /// Where the player has run over the last seconds, carried along with
    /// the camera so it stays on the grass he crossed, fading in from the
    /// oldest part and ending in an arrowhead at his feet.
    fn trail(
        &self,
        out: &mut Frame,
        set: &TrackSet,
        mark: &FootballMark,
        tracked: &football::TrackedBox,
        seconds: f64,
    ) {
        let now = (seconds * 1000.0).max(0.0) as u64;
        let back = (TRAIL_MS * mark.size.max(0.25)) as u64;
        let motion = self.motion(set);
        let mut points: Vec<(f32, f32)> = set
            .instants_of(mark.track)
            .into_iter()
            .filter(|(ms, _)| *ms + back >= now && *ms < now)
            .map(|(ms, then)| {
                let feet = pitch::feet(&then.bbox);
                let (u, v) = motion.carry(feet, ms, now).unwrap_or(feet);
                (u as f32, v as f32)
            })
            .collect();
        let feet = pitch::feet(&tracked.bbox);
        points.push((feet.0 as f32, feet.1 as f32));
        if points.len() < 2 {
            return;
        }
        // Four stretches, each more solid than the one behind it.
        let parts = 4.min(points.len() - 1);
        let per = (points.len() - 1) as f32 / parts as f32;
        for part in 0..parts {
            let from = (part as f32 * per).round() as usize;
            let to = (((part + 1) as f32 * per).round() as usize).min(points.len() - 1);
            if to <= from {
                continue;
            }
            let last = part + 1 == parts;
            self.line(
                out,
                points[from..=to].to_vec(),
                if last { draw::Kind::Arrow } else { draw::Kind::Line },
                mark.colour,
                mark.thickness as f32,
                false,
                mark.strength as f32 * (part + 1) as f32 / parts as f32,
            );
        }
    }

    /// A dashed line from the player's feet to the ball, green near, yellow
    /// within ten metres and red beyond, with the metres on it when the
    /// clip is calibrated; in the mark's colour and unlabelled when not.
    fn distance(
        &self,
        out: &mut Frame,
        set: &TrackSet,
        mark: &FootballMark,
        tracked: &football::TrackedBox,
        boxes: &HashMap<u32, football::TrackedBox>,
        seconds: f64,
    ) {
        let Some(ball) = boxes.get(&BALL) else {
            return;
        };
        let feet = pitch::feet(&tracked.bbox);
        let (bu, bv) = ball.centre();
        let ball_at = (f64::from(bu), f64::from(bv));
        let now = (seconds * 1000.0).max(0.0) as u64;
        let metres = self.to_pitch(set, now).and_then(|h| {
            let a = pitch::apply(&h, feet)?;
            let b = pitch::apply(&h, ball_at)?;
            Some((a.0 - b.0).hypot(a.1 - b.1))
        });
        let colour = match metres {
            Some(m) if m <= NEAR => [80, 220, 120],
            Some(m) if m <= FAR => [255, 210, 63],
            Some(_) => [229, 72, 77],
            None => mark.colour,
        };
        let a = (feet.0 as f32, feet.1 as f32);
        let b = (ball_at.0 as f32, ball_at.1 as f32);
        self.line(out, vec![a, b], draw::Kind::Line, colour, mark.thickness as f32, true, mark.strength as f32);
        if let Some(m) = metres {
            let middle = ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);
            label(out, &self.mapping, &format!("{m:.0} m"), middle, colour);
        }
    }

    /// The player's speed over the last second, in km/h, over his head:
    /// metres on the pitch, so only when the clip is calibrated.
    fn speed(
        &self,
        out: &mut Frame,
        set: &TrackSet,
        mark: &FootballMark,
        tracked: &football::TrackedBox,
        seconds: f64,
    ) {
        let now = (seconds * 1000.0).max(0.0) as u64;
        let Some(earlier) = set
            .instants_of(mark.track)
            .into_iter()
            .find(|(ms, _)| *ms + 1000 >= now && *ms + 250 < now)
        else {
            return;
        };
        let at = |ms: u64, bbox: &[f32; 4]| {
            let h = self.to_pitch(set, ms)?;
            pitch::apply(&h, pitch::feet(bbox))
        };
        let (Some(a), Some(b)) = (at(earlier.0, &earlier.1.bbox), at(now, &tracked.bbox)) else {
            return;
        };
        let elapsed = (now - earlier.0) as f64 / 1000.0;
        let kmh = (a.0 - b.0).hypot(a.1 - b.1) / elapsed * 3.6;
        if !kmh.is_finite() || kmh > 40.0 {
            return;
        }
        let [l, t, r, _] = tracked.bbox;
        let head = ((l + r) * 0.5, t - (tracked.bbox[3] - t) * 0.35);
        label(out, &self.mapping, &format!("{kmh:.0} km/h"), head, mark.colour);
    }

    /// A line across the whole pitch through the player's feet, the way
    /// the goal lines run: the last defender's line, the offside line.
    fn last_line(
        &self,
        out: &mut Frame,
        set: &TrackSet,
        mark: &FootballMark,
        tracked: &football::TrackedBox,
        seconds: f64,
    ) {
        let now = (seconds * 1000.0).max(0.0) as u64;
        let (Some(to_pitch), Some(to_picture)) = (self.to_pitch(set, now), self.to_picture(set, now))
        else {
            return;
        };
        let Some((x, _)) = pitch::apply(&to_pitch, pitch::feet(&tracked.bbox)) else {
            return;
        };
        let ends: Option<Vec<(f32, f32)>> = [0.0, PITCH.1]
            .into_iter()
            .map(|y| pitch::apply(&to_picture, (x, y)).map(|(u, v)| (u as f32, v as f32)))
            .collect();
        if let Some(ends) = ends {
            self.line(out, ends, draw::Kind::Line, mark.colour, mark.thickness as f32, false, mark.strength as f32);
        }
    }

    /// The five lanes and the three thirds on the grass, faint and dashed
    /// so the play stays the thing looked at.
    fn lanes(&self, out: &mut Frame, set: &TrackSet, mark: &FootballMark, seconds: f64) {
        let now = (seconds * 1000.0).max(0.0) as u64;
        let Some(to_picture) = self.to_picture(set, now) else {
            return;
        };
        let lanes = (1..5).map(|i| ((0.0, PITCH.1 * i as f64 / 5.0), (PITCH.0, PITCH.1 * i as f64 / 5.0)));
        let thirds = (1..3).map(|i| ((PITCH.0 * i as f64 / 3.0, 0.0), (PITCH.0 * i as f64 / 3.0, PITCH.1)));
        for (a, b) in lanes.chain(thirds) {
            let ends: Option<Vec<(f32, f32)>> = [a, b]
                .into_iter()
                .map(|p| pitch::apply(&to_picture, p).map(|(u, v)| (u as f32, v as f32)))
                .collect();
            if let Some(ends) = ends {
                self.line(
                    out,
                    ends,
                    draw::Kind::Line,
                    mark.colour,
                    (mark.thickness * 0.6) as f32,
                    true,
                    mark.strength as f32 * 0.6,
                );
            }
        }
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

/// The fonts a label is set in, read once: the bundled faces and the
/// system's.
fn fonts() -> &'static concat_text::Fonts {
    static FONTS: OnceLock<concat_text::Fonts> = OnceLock::new();
    FONTS.get_or_init(concat_text::Fonts::new)
}

/// A label set once and kept: its pixels, width and height.
type Painted = Arc<(Vec<u8>, u32, u32)>;

/// `text` on a plate of `colour`, centred on `at` (source fractions), at a
/// size that reads on the frame: a distance or a speed over the play.
fn label(out: &mut Frame, mapping: &Mapping, text: &str, at: (f32, f32), colour: [u8; 3]) {
    static PAINTED: OnceLock<Mutex<HashMap<(String, u32, [u8; 3]), Painted>>> = OnceLock::new();
    let height = ((out.height() as f32 * 0.045) as u32).max(18);
    let key = (text.to_owned(), height, colour);
    let cache = PAINTED.get_or_init(Mutex::default);
    let painted = {
        let held = cache.lock().ok().and_then(|held| held.get(&key).cloned());
        match held {
            Some(painted) => painted,
            None => {
                let light = 0.299 * f32::from(colour[0])
                    + 0.587 * f32::from(colour[1])
                    + 0.114 * f32::from(colour[2])
                    > 150.0;
                let style = concat_text::TitleStyle {
                    content: text.to_owned(),
                    font_family: "Hanken Grotesk".to_owned(),
                    font_size: 0.62,
                    font_weight: 700.0,
                    italic: false,
                    color: if light { "#101418" } else { "#ffffff" }.to_owned(),
                    align: concat_text::Align::Center,
                    stroke_width: 0.0,
                    stroke_color: "#000000".to_owned(),
                    shadow: false,
                    background: format!("#{:02x}{:02x}{:02x}e6", colour[0], colour[1], colour[2]),
                    background_radius: 0.0,
                    background_padding_x: 0.22,
                    background_padding_y: 0.06,
                    line_height: 1.0,
                    tracking: 0.0,
                    max_width: 0.0,
                    max_height: 0.0,
                };
                let Ok(rendered) = concat_text::render_frame(fonts(), &style, height * 6, height) else {
                    return;
                };
                let painted: Painted = Arc::new((rendered.rgba, rendered.width, rendered.height));
                if let Ok(mut held) = cache.lock() {
                    if held.len() > 256 {
                        held.clear();
                    }
                    held.insert(key, Arc::clone(&painted));
                }
                painted
            }
        }
    };
    let (rgba, w, h) = (&painted.0, painted.1 as i64, painted.2 as i64);
    let (x, y) = to_pixels(mapping, out, at);
    let (left, top) = (x as i64 - w / 2, y as i64 - h / 2);
    let (fw, fh) = (i64::from(out.width()), i64::from(out.height()));
    let pixels = out.pixels_mut();
    for row in 0..h {
        let py = top + row;
        if py < 0 || py >= fh {
            continue;
        }
        for col in 0..w {
            let px = left + col;
            if px < 0 || px >= fw {
                continue;
            }
            let from = ((row * w + col) * 4) as usize;
            let alpha = f32::from(rgba[from + 3]) / 255.0;
            if alpha <= 0.0 {
                continue;
            }
            let to = ((py * fw + px) * 4) as usize;
            for c in 0..3 {
                pixels[to + c] = (f32::from(pixels[to + c]) * (1.0 - alpha)
                    + f32::from(rgba[from + c]) * alpha
                    + 0.5) as u8;
            }
        }
    }
}

/// A source fraction to frame pixels, unclipped, through the clip's crop
/// and flips: the same way the drawings are placed.
fn to_pixels(mapping: &Mapping, frame: &Frame, (u, v): (f32, f32)) -> (f32, f32) {
    let [left, top, right, bottom] = mapping.crop;
    let x = (u - left) / (1.0 - left - right).max(1e-6);
    let y = (v - top) / (1.0 - top - bottom).max(1e-6);
    let x = if mapping.flip_h { 1.0 - x } else { x };
    let y = if mapping.flip_v { 1.0 - y } else { y };
    (x * frame.width() as f32, y * frame.height() as f32)
}

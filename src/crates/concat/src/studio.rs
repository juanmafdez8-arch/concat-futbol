// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! The window's state, and how it is published into Slint.
//!
//! Two kinds of state live here and the line between them is the whole
//! design:
//!
//! - **The edit** is the engine's. It lives in the open [`Session`], every
//!   change to it is a [`Command`], and this module only ever reads it back.
//!   A gesture in flight is previewed on an *echo* - a clone of the project
//!   the pointer mutates - and committed as one command on release, so undo
//!   undoes the drag and not a pixel of it.
//! - **The view** is the window's: selection, playhead, zoom, tool, the
//!   workspace's arrangement, which lanes are locked, what the dialogs show.
//!   None of it reaches the document.
//!
//! Everything Slint draws is produced by `publish` and its two halves from
//! those two, on every event that could have changed either.

#![allow(
    clippy::type_complexity,
    clippy::collapsible_if,
    clippy::manual_unwrap_or_default
)]

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use concat_effects::Catalogue;
use concat_effects::manifest::Kind as PackageKind;
use concat_effects::manifest::ParamType;
use concat_host::playback::ClipSpec;
use concat_host::{
    AnalyseRequest, Cutouts, EnhanceRequest, ProjectInfo, RegionRequest, Session, media, projects,
    templates,
};
use concat_media::{Pyramid, jpeg};
use concat_project::commands::{ClipMove, ClipPatch, TrackFlag, TrimEdge};
use concat_project::model::{
    self, AppliedFilter, Clip, Project, TextAlign, TextStyle, Timeline, Track, Transition,
};
use concat_project::{Command, why_not_merge};
use slint::{Model, ModelRc, SharedString, VecModel};

use crate::dock::{
    Dock, DockLayout, SEAT_GAP, default_dock, lay_out, nearest_row, row_at, row_top,
};
use crate::format::{
    WAVE_BAR, WAVE_PITCH, colour_of, frames_timecode, hex_of, hex_rgba, hex_with_alpha,
    wave_columns, wave_path, when_phrase,
};
use crate::grading;
use crate::host::{
    CachedStrip, Host, MediaArt, WindowArt, cached_media_art, cached_window_art, image_at,
    image_of, media_art, on_ui_in_project, spawn, spawn_art, spawn_in_project, spawn_strip,
    strip_window, window_art, window_span, window_start,
};
use crate::i18n::{self, t, tf};
use crate::panes::settings::installed;
use crate::platform;
use crate::prefs::Preferences;
use crate::presets::{self, TextPreset};
use crate::ui::*;

/// The monitor's output sizes, matching the picker's rows.
pub const OUTPUTS: [(i32, i32); 6] = [
    (1920, 1080),
    (3840, 2160),
    (1080, 1920),
    (1080, 1080),
    (1440, 1080),
    (2560, 1080),
];

/// Shortest clip the editor will make: a sixtieth of a second. Trims and
/// splits both floor at this, as the engine's own `MIN_CLIP_DURATION` does.
pub const MIN_DURATION: f32 = 1.0 / 60.0;

/// The three lane heights, in logical pixels.
///
/// Picture gets the tallest because a filmstrip is the one thing that needs
/// the room; sound the middle, where an envelope still has shape. The ladder
/// is what `TrackSize::Auto` picks from - see `lane_height`. Raised from
/// 80/60/40: with the name strip and the sound band taken off, a video's
/// frames had under forty pixels and a waveform under thirty, and both
/// read as crammed.
const LANE_LARGE: f32 = 108.0;
const LANE_MEDIUM: f32 = 80.0;
const LANE_SMALL: f32 = 44.0;

/// How long a title runs when it is placed: long enough to read, short
/// enough that trimming it is a nudge rather than a fight.
const LAYER_DURATION: f32 = 3.0;

/// One media item's filmstrip, as the lanes tile it.
pub struct Strip {
    /// Every sampled frame side by side.
    pub image: slint::Image,
    /// How many frames the picture holds.
    pub frames: i32,
    /// One frame's width in the picture's pixels.
    pub frame_width: i32,
    /// The picture's height in its own pixels.
    pub height: i32,
}

impl From<CachedStrip> for Strip {
    fn from(strip: CachedStrip) -> Self {
        Self {
            image: strip.image,
            frames: strip.frames as i32,
            frame_width: strip.frame_width as i32,
            height: strip.height as i32,
        }
    }
}

impl Strip {
    /// A decoded strip of `frames` frames.
    fn of(frame: &concat_core::frame::Frame, frames: u32) -> Self {
        Self {
            image: image_of(frame),
            frames: frames as i32,
            frame_width: (frame.width() / frames.max(1)) as i32,
            height: frame.height() as i32,
        }
    }
}

/// The key a cell's strip is held under: the media and the cell.
fn window_key(media_id: &str, level: u32, cell: u32) -> String {
    format!("{media_id}|{level}|{cell}")
}

/// How many cell strips are kept before the ones no clip on the current
/// timeline shows are let go. Each is a picture the size of the file's own
/// strip; a trim walks through several levels on its way down.
const WINDOWS_KEPT: usize = 96;

/// Steps per second the drawn waveform is quantised to. See `Studio::wave`:
/// it is what keeps a trim from synthesising a new envelope on every pointer
/// event, and the grid is fine enough that no step of it is visible.
const WAVE_STEPS: f32 = 30.0;

/// The export dialog's ladders, matching its rows.
/// The export sheet's resolution ladder, as the short side of the frame:
/// 4K, QHD, 1080p, 720p. The long side follows the project's own aspect,
/// so a 9:16 edit exports as 1080 x 1920 and a 21:9 one as 2520 x 1080 -
/// the picker chooses how fine, never which way round.
pub const EXPORT_SHORT_SIDES: [u32; 4] = [2160, 1440, 1080, 720];
/// Narrower than this, in logical pixels, and the workspace shows the
/// compact dock: a phone, a tablet held upright, a desktop window squeezed
/// to a corner. Four panes across less than this is four slivers.
pub const COMPACT_WIDTH: f32 = 860.0;
/// Megabits per second at 1080p30 for each quality tier, for the size
/// estimate; and the CRF each tier renders at.
pub(crate) const EXPORT_TIERS: [f32; 3] = [16.0, 8.0, 4.0];
pub(crate) const EXPORT_CRF: [u8; 3] = [16, 20, 26];
pub(crate) const AUDIO_BPS: f32 = 192_000.0;

/// The frame shapes the launch screen offers, as the ratio behind each
/// label: width over height.
///
/// A shape and a size rather than the four fixed frames this was. Those
/// offered one upright frame at one size and no square at all, so a phone
/// cut at 4K was not a thing the form could ask for. Every pair of indices
/// resolves through [`frame_size`], which is the one place that knows what
/// a label means in pixels.
pub const ASPECTS: [(&str, u32, u32); 4] = [
    ("16:9", 16, 9),
    ("9:16", 9, 16),
    ("1:1", 1, 1),
    ("4:3", 4, 3),
];

/// The sizes, as the *short* edge in pixels.
///
/// Which is what the "p" in 1080p has always counted, and the only reading
/// that survives turning a frame upright: 1080p landscape is 1920 x 1080
/// and 1080p vertical is 1080 x 1920, the same number of lines either way.
/// Naming the long edge instead would make a vertical 1080p a 1080 x 1920
/// frame at one moment and a 608 x 1080 frame at another.
pub const SIZES: [(&str, u32); 4] = [
    ("720p", 720),
    ("1080p", 1080),
    ("1440p", 1440),
    ("4K", 2160),
];

/// The frame an aspect and a size name, in pixels.
///
/// Both indices are clamped rather than trusted: they arrive from the form
/// as plain integers, and a frame is not worth a panic.
pub fn frame_size(aspect: usize, size: usize) -> (u32, u32) {
    let (_, w, h) = ASPECTS[aspect.min(ASPECTS.len() - 1)];
    let (_, short) = SIZES[size.min(SIZES.len() - 1)];
    // Multiplied before divided, so 1080 x 16 / 9 is 1920 and not 1080.
    let long = even(short * w.max(h) / w.min(h));
    if w >= h {
        (long, even(short))
    } else {
        (even(short), long)
    }
}

/// Rounded down to an even number of pixels.
///
/// Every ratio above already lands even at every size; this is for the next
/// one added. Chroma is subsampled by two in both directions in every format
/// the export writes, and an odd dimension is what an encoder either refuses
/// or quietly crops.
fn even(value: u32) -> u32 {
    value & !1
}

/// The frame rates a project is offered, on the launch sheet and the
/// project sheet alike, as exact fractions: 29.97 is 30000/1001 and never
/// anything else - a decimal is for reading, and export must never be
/// handed one. The rates cameras and screens run at, from a stop-motion
/// ten to a game capture's sixty; anything else is typed in as a custom
/// rate. https://github.com/jub0t/Concat/issues/199
pub const RATES: [(&str, i64, i64); 11] = [
    ("10", 10, 1),
    ("20", 20, 1),
    ("23.976", 24000, 1001),
    ("24", 24, 1),
    ("25", 25, 1),
    ("29.97", 30000, 1001),
    ("30", 30, 1),
    ("48", 48, 1),
    ("50", 50, 1),
    ("59.94", 60000, 1001),
    ("60", 60, 1),
];

/// The smallest and largest side a typed frame may have, in pixels. The
/// floor is a frame that still has a picture in it; the ceiling is 8K,
/// the texture size every GPU the compositor runs on can hold (the wgpu
/// floor), and past what any encoder the export drives will take.
pub const FRAME_SIDES: (u32, u32) = (16, 8192);

/// The slowest and fastest rate that may be typed, in frames a second.
pub const FRAME_RATES: (f64, f64) = (1.0, 240.0);

/// A typed frame as the project can hold it: each side within
/// [`FRAME_SIDES`] and even, rounded to the nearest even number rather
/// than down, so 1081 becomes 1082 and not a frame a line short.
pub fn custom_frame(width: f32, height: f32) -> (u32, u32) {
    let side = |value: f32| {
        let (low, high) = FRAME_SIDES;
        let value = if value.is_finite() { value } else { low as f32 };
        let nearest = ((value / 2.0).round() * 2.0) as i64;
        nearest.clamp(i64::from(low), i64::from(high)) as u32
    };
    (side(width), side(height))
}

/// A typed rate as the exact fraction a project stores. The NTSC rates are
/// the /1001 fractions cameras record at, so 29.97 and 23.976 come out as
/// 30000/1001 and 24000/1001 however they are rounded when typed; anything
/// else is kept to a thousandth, in lowest terms.
pub fn custom_rate(fps: f64) -> (i64, i64) {
    let (low, high) = FRAME_RATES;
    let fps = if fps.is_finite() {
        fps.clamp(low, high)
    } else {
        30.0
    };
    for whole in [24_i64, 30, 48, 60, 120, 240] {
        let ntsc = whole as f64 * 1000.0 / 1001.0;
        if (fps - ntsc).abs() < 0.006 {
            return (whole * 1000, 1001);
        }
    }
    let num = (fps * 1000.0).round() as i64;
    let divisor = gcd(num, 1000);
    (num / divisor, 1000 / divisor)
}

/// A rate as frames a second, for a field to show.
pub fn fps_of(num: i64, den: i64) -> f64 {
    num as f64 / den.max(1) as f64
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs().max(1)
    } else {
        gcd(b, a % b)
    }
}

/// What one effect library is showing: the words typed into it, the shelf
/// picked, and whether the star is down.
///
/// Held here rather than in the panel because the filtering is done here:
/// the model a library renders arrives already narrowed, which is what lets
/// it be laid out as a grid - a tile's place is its index, and an index only
/// means a place when every entry in the model is one that shows.
#[derive(Clone, Debug, Default)]
pub struct LibraryView {
    /// The search box's text.
    pub query: String,
    /// The chosen shelf, as an index into the kind's group list.
    pub group: i32,
    /// Only starred packages show, whatever shelf they are on.
    pub favourites: bool,
    /// Active category filter (e.g. "All", "Featured", "Retro & Film", etc.).
    pub category: String,
}

/// Which library a shelf index names. The numbers are the panel's own -
/// `PresetStack.shelf`, and `Library.view`'s shelf argument everywhere else -
/// and the order the tabs sit in.
pub const SHELF_KINDS: [PackageKind; 4] = [
    PackageKind::Filter,
    PackageKind::Effect,
    PackageKind::Audio,
    PackageKind::Transition,
];

/// `name` under the home directory, as a path string; empty when there is
/// no home to speak of, and the form then asks for a folder outright.
///
/// The platform's own answer, not `$HOME`: Windows does not set that, so
/// every folder the app suggested there came up empty. `name` is written
/// with `/` and joined a part at a time, so the separators match.
pub(crate) fn home_folder(name: &str) -> String {
    std::env::home_dir()
        .filter(|home| !home.as_os_str().is_empty())
        .map(|home| {
            name.split('/')
                .fold(home, |path, part| path.join(part))
                .to_string_lossy()
                .into_owned()
        })
        .unwrap_or_default()
}

/// The bottom-right notice: one at a time. The token is what the panel
/// watches, bumped per notice so the same sentence said twice blinks twice.
#[derive(Default)]
pub struct ToastState {
    pub token: i32,
    pub message: String,
    pub failed: bool,
}

/// Which edge of a clip a trim has hold of.
#[derive(Clone, Copy, PartialEq)]
pub enum Edge {
    Start,
    End,
}

/// Where one clip sat when a gesture began, so the whole set moves rigidly.
pub struct MoveOrigin {
    pub clip: String,
    pub start: f32,
    pub row: i32,
}

/// A pointer gesture in flight, previewed on the echo.
pub enum Gesture {
    None,
    Move {
        /// The clip actually grabbed; it is the one that snaps.
        primary: String,
        origins: Vec<MoveOrigin>,
        /// The lane heights as they were when the press landed. Frozen, not
        /// read live: an `Auto` lane resizes as clips land on it, which
        /// would move the very edges the next event measures against.
        lanes: Vec<f32>,
    },
    Trim {
        clip: String,
        edge: Edge,
        start: f32,
        duration: f32,
        source_start: f32,
    },
    /// Dragging a transition's duration handle on the timeline.
    TransitionResize {
        clip: String,
        original_duration: f32,
    },
    /// A drag on the stage: every selected picture under the playhead slides
    /// with the pointer, from where each one was when the press landed.
    StageMove {
        /// The picture grabbed; it is the one that snaps, and the rest
        /// follow it by the same amount.
        primary: String,
        origins: Vec<StageOrigin>,
        /// The press, in fractions of the frame.
        from: (f64, f64),
    },
    /// A corner grip dragged: the picture scales about its centre, by the
    /// ratio of the pointer's distance from that centre to what it was.
    StageScale {
        clip: String,
        scale: f64,
        /// The centre, in frame pixels — the unit the ratio is taken in, so
        /// a tall frame and a wide one scale at the same rate.
        centre: (f64, f64),
        from: f64,
        /// Half the picture's bounds at the press, in frame fractions, so
        /// an edge's position at any later scale is one multiplication.
        half: (f64, f64),
    },
    /// An edge grip pulls one axis: the picture's stretch across or down,
    /// about its centre, measured along the box's own axis so a turned
    /// picture stretches along itself and not along the frame.
    StageStretch {
        clip: String,
        /// The left and right edges pull the width; the others the height.
        across: bool,
        stretch: f64,
        centre: (f64, f64),
        /// The pointer's distance from the centre along the axis at the
        /// press, in frame pixels.
        from: f64,
        /// The picture's turn, to project the pointer onto its axes.
        rotation: f64,
    },
    /// A side grip on a title pulls the width its words wrap at, not the
    /// glyphs: the box edge follows the pointer and the words reflow into
    /// it. The width is twice the pointer's distance from the centre along
    /// the box's own axis, as the box is drawn about the centre.
    TextWidth {
        clip: String,
        centre: (f64, f64),
        /// The title's turn, to project the pointer onto its axis.
        rotation: f64,
    },
    /// The top or bottom grip on a title pulls the height of its box, on
    /// the same terms as [`Gesture::TextWidth`] pulls the width. Before
    /// this the two grips fell through to a stretch, and a caption pulled
    /// to fit came out with its glyphs squashed.
    /// https://github.com/jub0t/Concat/issues/119
    TextHeight {
        clip: String,
        centre: (f64, f64),
        rotation: f64,
    },
    /// The rotation grip dragged: the picture turns by the angle the pointer
    /// has swept about the centre since the press.
    StageRotate {
        clip: String,
        rotation: f64,
        centre: (f64, f64),
        from: f64,
    },
    /// A brush on the stage: the stroke so far, in source fractions for
    /// the command it becomes and in stage fractions for the line drawn
    /// under the pointer meanwhile.
    /// A line, an arrow or a curve being dragged out on the stage.
    Draw {
        clip: String,
        anchors: Vec<model::Anchor>,
        screen: Vec<(f32, f32)>,
    },
    Paint {
        clip: String,
        tool: model::BrushTool,
        size: f64,
        /// The source instant under the playhead when the stroke began.
        at: f64,
        points: Vec<[f64; 2]>,
        screen: Vec<(f32, f32)>,
    },
}

/// How the drawing tool draws: the style of the next drawing.
#[derive(Clone, Debug)]
pub struct DrawStyle {
    pub dashed: bool,
    pub colour: [u8; 3],
    pub thickness: f64,
    pub hold: f64,
    pub reveal: f64,
    pub bend: f64,
    /// How long Freeze and draw holds the frame, in seconds.
    pub freeze: f64,
}

impl Default for DrawStyle {
    fn default() -> Self {
        DrawStyle {
            dashed: false,
            colour: [0xff, 0xd4, 0x00],
            thickness: 1.0,
            hold: 4.0,
            reveal: 0.6,
            bend: 0.25,
            freeze: 5.0,
        }
    }
}

/// The custom cutout's brushes, in the inspector's order.
pub const BRUSHES: [model::BrushTool; 4] = [
    model::BrushTool::SmartBrush,
    model::BrushTool::Brush,
    model::BrushTool::SmartEraser,
    model::BrushTool::Eraser,
];

/// Where a picture was when a stage move began.
pub struct StageOrigin {
    pub clip: String,
    pub offset_x: f64,
    pub offset_y: f64,
}

/// A picture's footprint in the output frame, in fractions of it: where the
/// centre is, how much of the frame's width and height it covers before it
/// is turned, and the clockwise turn. The one place the monitor's box and
/// the compositor's quad agree, so the outline lands on the pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Footprint {
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
    pub rotation: f64,
}

impl Footprint {
    /// The half-extents of the turned box's axis-aligned bounds, in fractions
    /// of the frame: what snapping measures, since an edge of a turned
    /// picture is not a line the frame's edges can meet.
    pub fn half_bounds(&self, frame: (u32, u32)) -> (f64, f64) {
        let (width, height) = (f64::from(frame.0), f64::from(frame.1));
        let (sin, cos) = self.rotation.to_radians().sin_cos();
        let w = self.w * width;
        let h = self.h * height;
        (
            (w * cos.abs() + h * sin.abs()) / 2.0 / width,
            (w * sin.abs() + h * cos.abs()) / 2.0 / height,
        )
    }

    /// True when the frame point `(x, y)`, in fractions, is inside the turned
    /// box. `frame` is the output size, because the box is not square in
    /// pixels and the turn has to happen in pixels.
    pub fn contains(&self, x: f64, y: f64, frame: (u32, u32)) -> bool {
        let (width, height) = (f64::from(frame.0), f64::from(frame.1));
        let dx = (x - self.cx) * width;
        let dy = (y - self.cy) * height;
        let (sin, cos) = self.rotation.to_radians().sin_cos();
        // The inverse of the compositor's forward map: undo the clockwise
        // turn to land in the box's own frame, then it is an axis test.
        let lx = dx * cos + dy * sin;
        let ly = -dx * sin + dy * cos;
        lx.abs() <= self.w * width / 2.0 && ly.abs() <= self.h * height / 2.0
    }
}

/// A drag from the library, resolved against the timeline: the ghost the
/// lanes draw and the clip the release makes, as one answer.
#[derive(Clone)]
pub struct DropPlan {
    pub kind: ClipKind,
    pub label: String,
    /// The bin's media id, for a clip cut from a file. Empty for a title.
    pub media: String,
    pub start: f32,
    pub duration: f32,
    pub row: i32,
}

/// Every model the window is handed, kept for its lifetime rather than
/// rebuilt: a new model is a *reset*, and Slint answers a reset by dropping
/// every instance behind it - mid-gesture, that includes the TouchArea
/// holding the pointer.
pub struct Models {
    pub tabs: Rc<VecModel<TimelineTabData>>,
    pub tracks: Rc<VecModel<TrackData>>,
    pub clips: Rc<VecModel<ClipData>>,
    pub stage: Rc<VecModel<StageItemData>>,
    pub guides: Rc<VecModel<StageGuideData>>,
    pub media: Rc<VecModel<MediaItemData>>,
    pub video_effects: Rc<VecModel<EffectData>>,
    pub audio_effects: Rc<VecModel<EffectData>>,
    /// The catalogue's shelves, one list per kind, and the shelf labels.
    pub catalogue_effects: Rc<VecModel<CatalogueEntryData>>,
    pub catalogue_filters: Rc<VecModel<CatalogueEntryData>>,
    pub catalogue_audio: Rc<VecModel<CatalogueEntryData>>,
    pub catalogue_transitions: Rc<VecModel<CatalogueEntryData>>,
    pub effect_groups: Rc<VecModel<SharedString>>,
    pub filter_groups: Rc<VecModel<SharedString>>,
    pub audio_groups: Rc<VecModel<SharedString>>,
    pub transition_groups: Rc<VecModel<SharedString>>,
    /// The selected clip's two chains and their knobs.
    pub applied_visual: Rc<VecModel<AppliedEntryData>>,
    pub applied_audio: Rc<VecModel<AppliedEntryData>>,
    pub visual_params: Rc<VecModel<AppliedParamData>>,
    pub audio_params: Rc<VecModel<AppliedParamData>>,
    /// The colour panel's knobs.
    pub adjust_params: Rc<VecModel<AppliedParamData>>,
    /// The colour panel's wheels and curves, and the picture's chain's by
    /// link.
    pub adjust_wheels: Rc<VecModel<WheelData>>,
    pub adjust_curves: Rc<VecModel<CurveData>>,
    pub visual_wheels: Rc<VecModel<WheelGroupData>>,
    pub visual_curves: Rc<VecModel<CurveGroupData>>,
    /// The picture's chain's colour knobs, a row each.
    pub visual_colours: Rc<VecModel<ColourKnobData>>,
    /// A link's wheels and curves, and a curve's points, by the link - -1
    /// the colour panel - and the curve's key: kept like the rest, so a
    /// wheel or a point being dragged is not dropped when it is published
    /// again under the pointer.
    pub link_wheels: RefCell<HashMap<i32, Rc<VecModel<WheelData>>>>,
    pub link_curves: RefCell<HashMap<i32, Rc<VecModel<CurveData>>>>,
    pub curve_points: RefCell<HashMap<(i32, String), Rc<VecModel<CurvePointData>>>>,
    /// The keyframe cluster's rows and the libraries' views: synced like
    /// the rest, since a model handed over fresh is unequal to the last by
    /// identity and re-evaluates every binding on it.
    pub key_rows: Rc<VecModel<ClipKeyData>>,
    /// The Keyframes panel's rows; see `Studio::key_editor_rows`.
    pub key_editor_rows: Rc<VecModel<KeyRowData>>,
    /// The ruler's diamonds; see `Studio::key_marks`.
    pub key_marks: Rc<VecModel<f32>>,
    /// The plays on the ruler; see `Studio::montage_marks`.
    pub play_marks: Rc<VecModel<PlayMarkData>>,
    /// The tagging pane: its buttons, descriptors, list and filter.
    pub tag_buttons: Rc<VecModel<TagButtonData>>,
    pub tag_chips: Rc<VecModel<TagChipData>>,
    pub tag_rows: Rc<VecModel<TagRowData>>,
    pub tag_filter_options: Rc<VecModel<SharedString>>,
    pub tag_bars: Rc<VecModel<TagBarData>>,
    /// The evaluation: the library's profiles, and the reasons for the
    /// register in hand's verdict.
    pub eval_profiles: Rc<VecModel<SharedString>>,
    pub eval_reasons: Rc<VecModel<SharedString>>,
    /// The assistant's conversation.
    pub assistant_lines: Rc<VecModel<AssistantLineData>>,
    /// The pitch board's dots, and a calibration's points on it.
    pub pitch_dots: Rc<VecModel<PitchDotData>>,
    pub pitch_pairs: Rc<VecModel<PitchDotData>>,
    pub pitch_lines: Rc<VecModel<PitchLineData>>,
    pub pitch_kits: Rc<VecModel<slint::Color>>,
    /// The roles a player's card offers, none first.
    pub pitch_roles: Rc<VecModel<SharedString>>,
    pub library_views: Rc<VecModel<LibraryViewData>>,
    pub menu: Rc<VecModel<MenuItemData>>,
    pub bar: Rc<VecModel<MenuItemData>>,
    /// The sheets' option lists: what is installed, and who can speak.
    pub caption_models: Rc<VecModel<SharedString>>,
    pub speech_models: Rc<VecModel<SharedString>>,
    pub speech_model_details: Rc<VecModel<SharedString>>,
    pub speakers: Rc<VecModel<SharedString>>,
    pub speaker_details: Rc<VecModel<SharedString>>,
    pub speech_samples: Rc<VecModel<SharedString>>,
    pub speech_sample_waves: Rc<VecModel<SharedString>>,
    pub speech_sample_details: Rc<VecModel<SharedString>>,
    pub transcribers: Rc<VecModel<ModelData>>,
    pub voices: Rc<VecModel<ModelData>>,
    /// The Version page's releases, and beside each its day or that it
    /// is this one.
    pub versions: Rc<VecModel<SharedString>>,
    pub version_details: Rc<VecModel<SharedString>>,
    pub seats: Rc<VecModel<SeatBox>>,
    pub dividers: Rc<VecModel<DockDivider>>,
    pub recents: Rc<VecModel<RecentProjectData>>,
    /// The Text page's presets, published once from the loaded list.
    pub text_presets: Rc<VecModel<TextPresetData>>,
    /// The Latin ones, for the Text page's Styles shelf.
    pub style_presets: Rc<VecModel<TextPresetData>>,
    /// The rest, by the writing they are for, for its Languages shelf.
    pub language_presets: Rc<VecModel<TextPresetData>>,
    /// The families a title can be set in; see `Studio::font_families`.
    pub font_families: Rc<VecModel<SharedString>>,
    /// Those of them the font picker's search lets through.
    pub font_matches: Rc<VecModel<SharedString>>,
}

impl Models {
    pub fn new() -> Self {
        Self {
            tabs: Rc::new(VecModel::default()),
            tracks: Rc::new(VecModel::default()),
            clips: Rc::new(VecModel::default()),
            stage: Rc::new(VecModel::default()),
            guides: Rc::new(VecModel::default()),
            media: Rc::new(VecModel::default()),
            video_effects: Rc::new(VecModel::default()),
            audio_effects: Rc::new(VecModel::default()),
            catalogue_effects: Rc::new(VecModel::default()),
            catalogue_filters: Rc::new(VecModel::default()),
            catalogue_audio: Rc::new(VecModel::default()),
            catalogue_transitions: Rc::new(VecModel::default()),
            effect_groups: Rc::new(VecModel::default()),
            filter_groups: Rc::new(VecModel::default()),
            audio_groups: Rc::new(VecModel::default()),
            transition_groups: Rc::new(VecModel::default()),
            applied_visual: Rc::new(VecModel::default()),
            applied_audio: Rc::new(VecModel::default()),
            visual_params: Rc::new(VecModel::default()),
            audio_params: Rc::new(VecModel::default()),
            adjust_params: Rc::new(VecModel::default()),
            adjust_wheels: Rc::new(VecModel::default()),
            adjust_curves: Rc::new(VecModel::default()),
            visual_wheels: Rc::new(VecModel::default()),
            visual_colours: Rc::new(VecModel::default()),
            visual_curves: Rc::new(VecModel::default()),
            link_wheels: RefCell::new(HashMap::new()),
            link_curves: RefCell::new(HashMap::new()),
            curve_points: RefCell::new(HashMap::new()),
            key_rows: Rc::new(VecModel::default()),
            key_editor_rows: Rc::new(VecModel::default()),
            key_marks: Rc::new(VecModel::default()),
            play_marks: Rc::new(VecModel::default()),
            tag_buttons: Rc::new(VecModel::default()),
            tag_chips: Rc::new(VecModel::default()),
            tag_rows: Rc::new(VecModel::default()),
            tag_filter_options: Rc::new(VecModel::default()),
            eval_profiles: Rc::new(VecModel::default()),
            eval_reasons: Rc::new(VecModel::default()),
            assistant_lines: Rc::new(VecModel::default()),
            tag_bars: Rc::new(VecModel::default()),
            pitch_dots: Rc::new(VecModel::default()),
            pitch_pairs: Rc::new(VecModel::default()),
            pitch_lines: Rc::new(VecModel::default()),
            pitch_kits: Rc::new(VecModel::default()),
            pitch_roles: Rc::new(VecModel::default()),
            library_views: Rc::new(VecModel::default()),
            menu: Rc::new(VecModel::default()),
            bar: Rc::new(VecModel::default()),
            caption_models: Rc::new(VecModel::default()),
            speech_models: Rc::new(VecModel::default()),
            speech_model_details: Rc::new(VecModel::default()),
            speakers: Rc::new(VecModel::default()),
            speaker_details: Rc::new(VecModel::default()),
            speech_samples: Rc::new(VecModel::default()),
            speech_sample_waves: Rc::new(VecModel::default()),
            speech_sample_details: Rc::new(VecModel::default()),
            transcribers: Rc::new(VecModel::default()),
            voices: Rc::new(VecModel::default()),
            versions: Rc::new(VecModel::default()),
            version_details: Rc::new(VecModel::default()),
            seats: Rc::new(VecModel::default()),
            dividers: Rc::new(VecModel::default()),
            recents: Rc::new(VecModel::default()),
            text_presets: Rc::new(VecModel::default()),
            style_presets: Rc::new(VecModel::default()),
            language_presets: Rc::new(VecModel::default()),
            font_families: Rc::new(VecModel::default()),
            font_matches: Rc::new(VecModel::default()),
        }
    }
}

/// Republish a list into a live model without resetting it: rows that did
/// not change are not written, and the length changes a row at a time.
pub fn sync<T: Clone + PartialEq + 'static>(model: &VecModel<T>, next: Vec<T>) {
    let mut next = next;
    let shared = model.row_count().min(next.len());
    let tail = next.split_off(shared);
    for (row, value) in next.into_iter().enumerate() {
        if model.row_data(row).as_ref() != Some(&value) {
            model.set_row_data(row, value);
        }
    }
    while model.row_count() > shared {
        model.remove(model.row_count() - 1);
    }
    for value in tail {
        model.push(value);
    }
}

/// The window's state. See the module docs for what is whose.
pub struct Studio {
    pub host: Host,
    pub prefs: Preferences,
    /// What each effect library is showing, indexed the way `SHELF_KINDS`
    /// is: 0 filters, 1 effects, 2 audio.
    pub library: [LibraryView; 4],

    // ── the edit ──
    pub session: Option<Session>,
    /// A clone of the project a gesture is mutating. `project()` reads it
    /// while it exists; a command replaces it.
    pub echo: Option<Project>,
    /// Stands in for a project when none is open, so every reader has
    /// something to read.
    empty: Project,
    /// Unsaved changes, and the timer that writes them.
    dirty: bool,
    autosave: slint::Timer,

    // ── the bin ──
    pub media: crate::panes::media_bin::MediaBin,
    /// Decoded art by media id, and the ids a worker is decoding for.
    pub peaks: HashMap<String, Arc<Pyramid>>,
    /// Filmstrips by media id: the picture, how many frames are in it, one
    /// frame's width and the strip's height, in the picture's own pixels.
    pub strips: HashMap<String, Strip>,
    /// Filmstrips of one cell of a file each, by `window_key`, for the cuts
    /// too short a piece of their footage for the file's strip to show as
    /// more than one frame repeated. See `host::strip_window`.
    pub windows: HashMap<String, Strip>,
    art_pending: HashSet<String>,
    /// Art keys a decode came back empty for: not asked again this
    /// project, or a file that cannot be read would be decoded on every
    /// event.
    art_failed: HashSet<String>,
    /// What the last art scan was for: the document's revision, the
    /// screen, and how much was pending. The scan runs when one of them
    /// changes and is a comparison on the pointer stream between.
    art_stamp: Option<(u64, bool, usize)>,
    /// Media whose proxy has been seen to, this project: `proxy::ensure`
    /// costs two stats of the file, and once is enough.
    proxied: HashSet<String>,
    window_pending: HashSet<String>,
    /// Envelopes, keyed by the things they are computed from. A move
    /// changes none of them, and a publish happens on every frame of one.
    waves: RefCell<HashMap<String, SharedString>>,

    // ── the view ──
    /// The timeline's view: scroll, zoom, tool, and what the lanes know
    /// about a track that the document does not.
    pub lanes: crate::panes::timeline::TimelinePane,
    pub selection: Vec<String>,
    pub playhead: f32,
    pub playing: bool,
    transport: slint::Timer,
    /// The preview axis: the instant under the pointer while it crosses the
    /// lanes, which the monitor shows instead of the playhead's. None when
    /// the pointer is elsewhere, or the axis is off, or the cut is playing.
    pub hover: Option<f32>,
    /// Stops the burst of sound a hover with audio plays.
    hover_hush: slint::Timer,
    /// One clip, held for Paste.
    pub clipboard: Vec<Held>,
    /// The monitor: its frame, and the requests for the next.
    pub monitor: crate::panes::monitor::MonitorPane,

    // ── the sheets and menus ──
    pub export: crate::panes::export::ExportPane,
    pub settings: crate::panes::settings::SettingsPane,
    pub relink: crate::panes::relink::RelinkPane,
    /// What the confirmation sheet is asking about, while it is up.
    pub confirm: Option<Confirm>,
    /// The clip whose incoming transition is selected, while no clip is:
    /// a transition is picked on its own, not as part of the clip it rides
    /// into. Read through [`Studio::selected_transition`], which holds it
    /// to an empty clip selection and a clip that still has one.
    pub transition_selected: Option<String>,
    pub open_menu: i32,
    pub menu_bar_token: i32,
    pub menu_target: Option<String>,
    /// The media the bin's menu was opened on, when it was the bin's menu
    /// and not a clip's: the two share one set of rows and one token.
    pub menu_media: Option<String>,
    pub menu_token: i32,
    /// The run of one kind of continuous message being folded into one
    /// line of the activity trail; see `note_activity`.
    activity_burst: Option<crate::panes::Burst>,
    pub toast: ToastState,

    // ── the launch screen ──
    pub on_start: bool,
    pub start: crate::panes::start::StartPane,
    pub recents: Vec<ProjectInfo>,
    pub posters: HashMap<String, slint::Image>,
    posters_pending: HashSet<String>,
    pub project_name: String,

    // ── the workspace ──
    pub dock: Dock,
    /// The dock not showing: the compact one while the window is wide,
    /// the wide one while it is compact. Swapped with `dock` as the
    /// window crosses `COMPACT_WIDTH`, so each keeps its shape.
    pub dock_aside: Dock,
    /// Whether `dock` is the compact one.
    pub compact: bool,
    pub workspace: (f32, f32),
    pub divider_press: Option<(usize, f32, f32)>,
    pub gesture: Gesture,
    /// The snap lines of a stage move in flight; empty between moves.
    pub stage_guides: Vec<StageGuideData>,
    /// Where the inspector should go, and a count that changes every time
    /// something is applied from the library; see `Editor.inspector-jump-token`.
    pub inspector_jump: (i32, &'static str, &'static str),
    /// Bumped when the library should show its Media page: a file the
    /// editor just made and put in the bin. See `Editor.media-jump-token`.
    pub media_jump: i32,
    /// A look being shown before it is laid down: the filter's id. It is
    /// drawn into the frames the monitor asks for as a layer over the whole
    /// picture, and nowhere else - the timeline does not have it until the
    /// card is double-clicked or its plus pressed, which lays a filter
    /// layer at the playhead.
    audition: Option<String>,
    /// Counts every change to the document. What the flattened clip list
    /// below is keyed on, so a frame of an unchanged document reuses it.
    revision: u64,
    /// The last flattening of the document for the monitor - titles
    /// included - and the revision and output size it was made at. Shared
    /// with the monitor by pointer, so it can keep its plan for as long as
    /// the list is the same one.
    flat: Option<(
        u64,
        u32,
        u32,
        std::sync::Arc<Vec<concat_export::ExportClip>>,
    )>,
    /// An inspector commit waiting to land: a knob being dragged commits
    /// on every move, and each commit was a command, an undo of the last,
    /// a rebuild of the mix and a full publish. The commit is held until
    /// the moves pause; the echo shows the value meanwhile.
    commit_pending: bool,
    /// The clip the held commit is for: the one that was selected when the
    /// inspector wrote to the echo. Kept apart from the selection because
    /// the two can come apart before the commit lands - a press on the
    /// lanes' floor blurs a title's text box, which holds the commit, and
    /// then clears the selection on the release - and a commit that looked
    /// the clip up in the selection at landing time found nothing and
    /// dropped the words with the echo.
    /// The clips the echo was written for; see `commit_now`. Several when
    /// the inspector edits every selected title at once.
    commit_targets: Vec<String>,
    commit_timer: slint::Timer,
    /// What the catalogue shelves were last built from; while nothing in
    /// it changes the shelves are not rebuilt.
    shelf_stamp: std::cell::RefCell<Option<ShelfStamp>>,
    /// Every card still on the shelves, by package id, loaded once - an
    /// author's own `preview.png`, else the card drawn for the package -
    /// and kept: the shelves are rebuilt on every publish and a picture
    /// read from disk each time would be the slowest thing in the window.
    /// A card not drawn yet is kept as no picture until it arrives.
    look_art: std::cell::RefCell<HashMap<String, slint::Image>>,
    /// Where each picture package's card is kept, by package id; see
    /// `concat_host::cards`. Planned whenever the catalogue loads.
    card_paths: HashMap<String, std::path::PathBuf>,
    /// How many cards have arrived since the window opened: part of the
    /// shelf stamp, so a card that lands rebuilds the shelves.
    cards_drawn: u64,
    /// Whether cards are being drawn, and whether the catalogue changed
    /// while they were, which makes for another round when this one ends.
    cards_busy: bool,
    cards_again: bool,
    /// The watch on the user's effects folder: a poll every couple of
    /// seconds of what is in it, cheaper than a file-system watcher and
    /// one dependency fewer, at a cost nobody will notice.
    packages_watch: slint::Timer,
    /// The folder as it was when the catalogue was last built, and, once
    /// a change has been seen, the folder as it was at that sight: the
    /// catalogue is rebuilt only when two polls agree, so a package still
    /// being copied in is not loaded half-way.
    packages_seen: u64,
    packages_pending: Option<u64>,
    /// The last inspector commit: what it changed and when. A control that
    /// is dragged commits on every move, and each of those would be an undo
    /// step of its own; a commit that changes the same thing as the last
    /// one, within a moment of it, replaces it in the history instead.
    last_commit: Option<(String, std::time::Instant)>,
    /// Each title's painted block in frame pixels, by clip id, as of the
    /// last time the monitor asked for a frame. What the stage box for a
    /// text clip is drawn from; see `footprint`.
    /// Per text clip: the painted block's size in frame pixels, and its
    /// centre's offset from the clip's centre - see `TitleClip::offset`.
    #[allow(clippy::type_complexity)]
    pub title_blocks: HashMap<String, ((u32, u32), (i32, i32))>,
    pub drop: Option<DropPlan>,
    pub project_sheet: crate::panes::project::ProjectPane,
    pub captions: crate::panes::captions::CaptionsPane,
    pub speech: crate::panes::speech::SpeechPane,
    /// A file from the library lent the monitor and the speakers; see
    /// `crate::source`.
    pub source: Option<crate::source::Source>,
    /// The VU meters' feed; see `crate::meters`.
    pub meters: crate::meters::MeterFeed,
    /// The voiceover take, while one runs.
    pub voiceover: crate::panes::voiceover::VoiceoverPane,
    /// Every speaker the voice engine offers, in its own order.
    /// The looks the Text page offers; see `presets`.
    pub text_presets: Vec<TextPreset>,
    /// The fonts in the app's own folder - imported from the Text
    /// inspector, or brought by a preset - as (family, file), read once at
    /// start and added to as files come in. Offered to every title; a
    /// project a title is set in one of them registers the file so the
    /// painter finds it. See `presets::installed_fonts`.
    pub installed_fonts: Vec<(String, String)>,
    /// The families the machine has, found on a worker after the window is
    /// up - the system's fonts take a moment to read - and empty until then.
    pub system_fonts: Vec<String>,
    /// A fingerprint of what `font_families` was last published from, so
    /// a publish that changes none of it leaves the picker's list alone.
    fonts_published: std::cell::Cell<Option<u64>>,
    /// What the font picker's search box holds.
    pub font_query: String,
    /// The families and the query the search's matches were last
    /// published from, hashed.
    font_search_published: std::cell::Cell<Option<u64>>,

    /// The languages Settings › General offers, in its order; see `i18n`.
    pub languages: Vec<i18n::Language>,

    // ── cutouts ──
    /// The brush a custom cutout paints with: a row of `BRUSHES`.
    pub brush: usize,
    /// The brush's diameter, as a fraction of the picture's width.
    pub brush_size: f64,
    /// Whether a press on the stage paints the selected custom cutout
    /// rather than moving the picture.
    pub painting: bool,
    /// The mask analyses running, by media id, and how far each has got.
    cutout_jobs: HashMap<String, (bool, f32)>,

    // ── drawing ──
    /// The drawing tool out, as a row of `DRAW_KINDS`, if one is.
    pub draw_tool: Option<usize>,
    /// The style the next drawing takes.
    pub draw_style: DrawStyle,
    /// A zone or a link being clicked out: its clip, its points, and where
    /// they were on the stage.
    draw_draft: Option<(String, Vec<model::Anchor>, Vec<(f32, f32)>)>,

    // ── the analysis toolbar ──
    /// The tool out on the monitor's toolbar; see `analysis_tool_pick`.
    pub analysis_tool: i32,
    /// The toolbar's colour, as a row of `MARK_COLOURS`.
    pub analysis_colour: usize,
    /// A player tool's press kept while the players are being found: the
    /// clip, the source point, the source instant and the mark.
    football_pending: Option<(String, f32, f32, f64, model::MarkStyle)>,
    /// The dock the analysis workspace replaced, while it is showing.
    dock_before_analysis: Option<Dock>,
    /// A pitch calibration being made: its clip, its source instant, its
    /// pairs `[u, v, x, y]`, and a video point waiting for its landmark.
    calibrating: Option<(String, u64, Vec<[f64; 4]>, Option<(f64, f64)>)>,
    /// The pitch board's camera motion and calibration, keyed by the
    /// files they were read from.
    board_cache: RefCell<
        Option<(
            (
                std::path::PathBuf,
                Option<std::time::SystemTime>,
                Option<std::time::SystemTime>,
                Option<std::time::SystemTime>,
            ),
            Arc<concat_vision::football::pitch::Motion>,
            concat_vision::football::pitch::Calibration,
            Arc<HashMap<u32, (i32, String, i32)>>,
        )>,
    >,

    // ── tagging ──
    /// The register in hand: the last one made or picked, by id.
    tag_current: Option<String>,
    /// The evaluation's profile in use, by its place in the library.
    eval_profile: usize,
    /// The team the evaluation judges: 1 or 2, as the analysis split kits.
    eval_team: u8,
    /// The assistant's conversation as shown: 0 the coach, 1 the assistant,
    /// 2 a failure; and as the model reads it.
    assistant_lines: Vec<(u8, String)>,
    assistant_history: Vec<crate::assistant::Turn>,
    /// A question with the model.
    assistant_busy: bool,
    /// Which model answered last: "local" or "claude".
    assistant_provider_used: String,
    /// Library clips waiting for the player analysis, by clip id.
    library_queue: Vec<String>,
    /// The microphone, while the coach is talking to the assistant.
    assistant_recorder: Option<concat_host::record::Recorder>,
    /// What was said, being written out.
    assistant_hearing: bool,
    /// The assistant's settings, read once and again after each save.
    assistant_config: RefCell<Option<crate::assistant::Config>>,
    /// The coach's library, keyed by its files' times.
    #[allow(clippy::type_complexity)]
    eval_library_cache: RefCell<Option<([Option<std::time::SystemTime>; 4], Arc<EvalLibrary>)>>,
    /// A clip's analysis read onto the pitch, keyed by its files' times.
    #[allow(clippy::type_complexity)]
    eval_context_cache: RefCell<
        Option<(
            (
                std::path::PathBuf,
                Option<std::time::SystemTime>,
                Option<std::time::SystemTime>,
                Option<std::time::SystemTime>,
            ),
            Arc<concat_vision::football::eval::Context>,
        )>,
    >,
    /// The category the tagging pane's list shows, by name; all when none.
    tag_filter: Option<String>,
    /// The fewest stars a register needs to be listed.
    tag_min_rating: u8,
    /// Whether the tagging pane shows the template editor.
    tag_editing: bool,

    // ── montage ──
    /// The in and out marks, in timeline seconds; see `mark_in_here`.
    pub mark_in: Option<f32>,
    pub mark_out: Option<f32>,

    // ── football ──
    /// Whether a press on the stage picks a tracked player of the selected
    /// clip rather than moving the picture.
    pub football_picking: bool,
    /// The track picked last, which the marks are put on.
    pub football_chosen: Option<u32>,
    /// The football job running: what it is, for which clip, how far.
    pub football_job: Option<(String, String, f32)>,
    /// The players whose silhouettes have been asked for since the last
    /// analysis, by media file; see `ensure_football_silhouettes`.
    football_tried: HashSet<(String, u32)>,
    /// Enhance at work, by clip id: whether the model is still
    /// downloading, and how far along. One at a time, like every long job.
    enhance_jobs: HashMap<String, (bool, f32)>,
    /// Reverse writing a clip's backwards copy, by clip: how far along.
    reverse_jobs: HashMap<String, f32>,
    /// The smart stroke being read, by the same name, while one is.
    region_job: Option<String>,
    /// The last smart stroke as the stage drew it, kept on screen from the
    /// release until the model has read what was under it, so the paint
    /// does not vanish before its answer arrives.
    pending_stroke: Option<(String, f32, bool)>,
}

// ── conversions between the document and the window ─────────────────────

/// The key a media item's art is kept under: the media id for its pictures
/// and its default stream's waveform, and the id with the stream for the
/// waveform of another stream a clip plays. Peaks and pending jobs share it.
fn art_key(media_id: &str, stream: Option<u32>) -> String {
    match stream {
        None => media_id.to_owned(),
        Some(index) => format!("{media_id}#{index}"),
    }
}

/// One audio track's row in the Audio panel's list: the name the file gave
/// it or its number, and its channel layout.
fn audio_track_label(track: &model::AudioTrack, position: usize) -> String {
    let name = if track.title.is_empty() {
        tf("studio.track", &[&(position + 1)])
    } else {
        track.title.clone()
    };
    let layout = match track.channels {
        1 => t("studio.mono"),
        2 => t("studio.stereo"),
        channels => tf("studio.channels", &[&channels]),
    };
    format!("{name} · {layout}")
}

fn kind_of(clip: &Clip) -> ClipKind {
    match clip.kind {
        model::ClipKind::Video => ClipKind::Video,
        model::ClipKind::Audio => ClipKind::Audio,
        model::ClipKind::Image => ClipKind::Image,
        model::ClipKind::Text => ClipKind::Text,
        model::ClipKind::Layer => ClipKind::Filter,
        model::ClipKind::Shape => ClipKind::Shape,
    }
}

fn align_of(align: TextAlign) -> TextAlignment {
    match align {
        TextAlign::Left => TextAlignment::Left,
        TextAlign::Center => TextAlignment::Center,
        TextAlign::Right => TextAlignment::Right,
    }
}

/// A title as it is first placed: the embedded face, centred, white.
fn new_title_style(prefs: &crate::prefs::Preferences) -> TextStyle {
    // The look a title was kept as, when one was (see
    // `Studio::use_title_style_for_new`); the bundled face otherwise.
    let base = prefs.title_style.clone().unwrap_or_else(|| TextStyle {
        font_family: "Hanken Grotesk".to_owned(),
        font_weight: 600.0,
        ..TextStyle::default()
    });
    TextStyle {
        content: "New title".to_owned(),
        ..base
    }
}

/// A label for a catalogue id: "black-white" reads as "Black White".
/// One kind's shelves for the inspector: the shelf labels, in catalogue
/// order, and every package on them with the index of its shelf.
fn shelves(
    kind: PackageKind,
    view: &LibraryView,
    favourites: &[String],
    look_art: &std::cell::RefCell<HashMap<String, slint::Image>>,
    card_paths: &HashMap<String, std::path::PathBuf>,
) -> (Vec<SharedString>, Vec<CatalogueEntryData>) {
    let mut groups: Vec<String> = Vec::new();
    let mut entries = Vec::new();
    let query = view.query.trim().to_lowercase();
    for package in Catalogue::builtin().of_kind(kind) {
        let meta = &package.manifest.effect;
        // The colour panel's own package: every picture clip can carry it,
        // and it has a tab, not a shelf.
        if meta.id == ADJUST_ID {
            continue;
        }
        let category = if meta.category.is_empty() {
            "Other".to_owned()
        } else {
            meta.category.clone()
        };
        // Shelf and package names come from the manifests in English and
        // are looked up like any other string, so a locale can carry them.
        let group = match groups.iter().position(|held| *held == category) {
            Some(index) => index,
            None => {
                groups.push(category.clone());
                groups.len() - 1
            }
        };
        // The shelves are built from every package and only the entries are
        // narrowed: a strip that lost a segment because a search matched
        // nothing on it would move under the pointer as you typed.
        let name = i18n::package_text(&meta.id, "name", &meta.name);
        let description = i18n::package_text(&meta.id, "description", &meta.description);
        let starred = favourites.iter().any(|held| held == &meta.id);
        let shown = if view.favourites {
            starred
        } else if !query.is_empty() {
            // Across every shelf while a query is live. Someone typing
            // "echo" wants the echo, not the echo that happens to be filed
            // where they were already looking.
            name.to_lowercase().contains(&query) || description.to_lowercase().contains(&query)
        } else if !view.category.is_empty() {
            package.matches_category(&view.category)
        } else if view.group >= 0 {
            group as i32 == view.group
        } else {
            true
        };
        if !shown {
            continue;
        }
        // An author's own still from the package's folder, else the card
        // drawn for the package by its own shader; read once and kept.
        let art = look_art
            .borrow_mut()
            .entry(meta.id.clone())
            .or_insert_with(|| {
                package
                    .folder
                    .as_deref()
                    .and_then(|folder| {
                        slint::Image::load_from_path(&folder.join("preview.png")).ok()
                    })
                    .or_else(|| {
                        let card = card_paths.get(&meta.id)?;
                        slint::Image::load_from_path(card).ok()
                    })
                    .unwrap_or_default()
            })
            .clone();
        entries.push(CatalogueEntryData {
            id: meta.id.as_str().into(),
            name: name.into(),
            category: i18n::shelf_text("categories", &category).into(),
            group: group as i32,
            description: description.into(),
            favourite: starred,
            art,
        });
    }
    (
        groups
            .into_iter()
            .map(|group| SharedString::from(i18n::shelf_text("categories", &group)))
            .collect(),
        entries,
    )
}

/// How a manifest's unit is read out. Anything the inspector has no words
/// for is a plain number.
fn format_of(unit: &str) -> ParamFormat {
    match unit.trim() {
        "%" => ParamFormat::Percent,
        "dB" => ParamFormat::Decibels,
        "Hz" => ParamFormat::Hertz,
        "s" => ParamFormat::Seconds,
        "ms" => ParamFormat::Millis,
        "st" => ParamFormat::Pitch,
        "K" => ParamFormat::Kelvin,
        "EV" => ParamFormat::Stops,
        "°" => ParamFormat::Degrees,
        "x" => ParamFormat::Rate,
        _ => ParamFormat::Plain,
    }
}

/// What a batch of inspector commands touches, as a string two commits can
/// be compared by: the variant names, and for a patch the fields it sets.
fn commit_key(commands: &[Command]) -> String {
    commands
        .iter()
        .map(|command| match command {
            Command::SetClipTransform { .. } => "transform".to_owned(),
            Command::SetClipCutout { .. } => "cutout".to_owned(),
            Command::SetClipSpeed { .. } => "speed".to_owned(),
            Command::SetClipSpeedCurve { .. } => "curve".to_owned(),
            Command::UpdateClip { patch, .. } => {
                let mut fields = Vec::new();
                if patch.name.is_some() {
                    fields.push("name");
                }
                if patch.volume.is_some() {
                    fields.push("volume");
                }
                if patch.fade_in.is_some() {
                    fields.push("fade_in");
                }
                if patch.fade_out.is_some() {
                    fields.push("fade_out");
                }
                if patch.opacity.is_some() {
                    fields.push("opacity");
                }
                if patch.text.is_some() {
                    fields.push("text");
                }
                if patch.video_effects.is_some() {
                    fields.push("video_effects");
                }
                if patch.filters.is_some() {
                    fields.push("filters");
                }
                if patch.crop.is_some() {
                    fields.push("crop");
                }
                format!("patch:{}", fields.join("+"))
            }
            _ => "other".to_owned(),
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// The field a keyable property is set through - the inverse of
/// `Studio::key_property_of`, for handing a row back to the panel.
fn key_field_of(property: model::KeyProperty) -> ClipField {
    match property {
        model::KeyProperty::Scale => ClipField::Scale,
        model::KeyProperty::OffsetX => ClipField::OffsetX,
        model::KeyProperty::OffsetY => ClipField::OffsetY,
        model::KeyProperty::Rotation => ClipField::Rotation,
        model::KeyProperty::Opacity => ClipField::Opacity,
        model::KeyProperty::Volume => ClipField::Volume,
    }
}

/// What the catalogue shelves are a function of; see `Studio::shelf_stamp`.
#[derive(PartialEq)]
struct ShelfStamp {
    catalogue: usize,
    cards: u64,
    lang: String,
    views: Vec<(String, i32, bool, String)>,
    favourites: Vec<String>,
}

/// The built-in colour package's id; see `adjust_rows` and `Studio::adjust_set`.
const ADJUST_ID: &str = "concat.adjust";

/// The ease a key put on at `at` inherits: that of whichever key it joins
/// behind, so laying a run of keys down does not alternate between shapes.
/// The first key on a parameter has nothing to inherit and gets the
/// straight line.
fn ease_before(keys: &[model::ParamKey], at: f64) -> model::KeyEase {
    keys.iter()
        .rev()
        .find(|key| key.at < at)
        .map_or(model::KeyEase::LINEAR, |key| key.ease)
}

/// The playhead's place in `clip`, `0..=1`, held to its ends: where a keyed
/// property is shown and written from. `Studio::key_point` is the same
/// without the clamp, for the diamonds, which must not land a key on an end
/// nobody aimed at.
fn place_in(clip: &Clip, playhead: f32) -> f64 {
    if clip.duration <= 0.0 {
        return 0.0;
    }
    ((f64::from(playhead) - clip.start) / clip.duration).clamp(0.0, 1.0)
}

/// What a keyable property is worth on screen: its constant, or once it
/// carries keys, its ride's value at `at`.
fn shown(clip: &Clip, property: model::KeyProperty, at: f64) -> f64 {
    if clip.is_keyed(property) {
        clip.value_at(property, at)
    } else {
        clip.constant(property)
    }
}

/// A keyable property's value in the knob's units: the model's, except
/// the level, which the knob shows in decibels and the clip holds as gain.
fn knob_value(property: model::KeyProperty, value: f64) -> f64 {
    match property {
        model::KeyProperty::Volume => {
            if value <= 0.001 {
                -60.0
            } else {
                (20.0 * value.log10()).clamp(-60.0, 24.0)
            }
        }
        _ => value,
    }
}

/// The reverse of [`knob_value`]: a knob's number as the model holds it.
fn model_value(property: model::KeyProperty, knob: f64) -> f64 {
    match property {
        model::KeyProperty::Volume => {
            if knob <= -60.0 {
                0.0
            } else {
                10f64.powf(knob / 20.0)
            }
        }
        _ => knob,
    }
}

/// The range a keyable knob runs over, in the knob's units: what the
/// curve view's height spans. The same numbers as the inspector's rows.
fn knob_range(property: model::KeyProperty) -> (f64, f64) {
    match property {
        model::KeyProperty::Scale => (0.05, 8.0),
        model::KeyProperty::OffsetX | model::KeyProperty::OffsetY => (-1.0, 1.0),
        model::KeyProperty::Rotation => (-180.0, 180.0),
        model::KeyProperty::Opacity => (0.0, 1.0),
        model::KeyProperty::Volume => (-60.0, 24.0),
    }
}

/// The keyable property a batch edits relative to the first clip, for the
/// fields that place a picture; None for every field a batch sets outright.
fn relative_property(field: ClipField) -> Option<model::KeyProperty> {
    match field {
        ClipField::OffsetX => Some(model::KeyProperty::OffsetX),
        ClipField::OffsetY => Some(model::KeyProperty::OffsetY),
        ClipField::Rotation => Some(model::KeyProperty::Rotation),
        ClipField::Scale => Some(model::KeyProperty::Scale),
        _ => None,
    }
}

/// What a batch's knob means for one of its clips. The knob shows the
/// first clip's value, `reference`, and the others keep their distance
/// from it: a position or a turn moves by the amount the knob moved, a
/// scale by the factor, so a row of titles slides together rather than
/// landing on one spot. The first clip itself lands on the knob's value.
fn batch_value(field: ClipField, value: f64, reference: f64, current: f64) -> f64 {
    match field {
        ClipField::OffsetX | ClipField::OffsetY | ClipField::Rotation => {
            current + (value - reference)
        }
        ClipField::Scale if reference > 0.0 => current * (value / reference),
        _ => value,
    }
}

/// Writes a keyable property the way the person means it: the constant
/// while the property has no keys, and once it rides, the key at `at` -
/// put on if there was none, with the ease of the key behind it. The engine
/// stops reading the constant the moment a property is keyed, so a knob or
/// a stage drag that wrote it would change nothing on screen.
fn write_keyable(clip: &mut Clip, property: model::KeyProperty, value: f64, at: f64) {
    if clip.is_keyed(property) {
        let ease = clip
            .keys_on(property)
            .rfind(|key| key.at < at)
            .map_or(model::KeyEase::LINEAR, |key| key.ease);
        clip.set_key(property, at, value, ease);
        return;
    }
    match property {
        model::KeyProperty::Scale => clip.scale = value,
        model::KeyProperty::OffsetX => clip.offset_x = value,
        model::KeyProperty::OffsetY => clip.offset_y = value,
        model::KeyProperty::Rotation => clip.rotation = value,
        model::KeyProperty::Opacity => clip.opacity = value,
        model::KeyProperty::Volume => clip.volume = value,
    }
}

/// A command as the activity trail names it: its kind, and for a batch
/// the kinds inside it. Never its contents - a patch carries a title's
/// words - which is what keeps a line a line.
fn command_name(command: &Command) -> String {
    if let Command::Batch { commands } = command {
        let inside: Vec<String> = commands.iter().map(command_name).collect();
        return format!("Batch[{}]", inside.join(", "));
    }
    let text = format!("{command:?}");
    text.chars()
        .take_while(|c| c.is_ascii_alphanumeric())
        .collect()
}

/// The commands that turn `before` into `after`, clip `id` as the echo
/// left it against the document's: a transform, a cutout, a speed, the
/// keys, and one patch for everything else that differs. Empty when
/// nothing does.
fn clip_commands(id: &str, before: &Clip, after: &Clip) -> Vec<Command> {
    let mut commands = Vec::new();
    if after.scale != before.scale
        || after.offset_x != before.offset_x
        || after.offset_y != before.offset_y
        || after.rotation != before.rotation
        || after.stretch_x != before.stretch_x
        || after.stretch_y != before.stretch_y
    {
        commands.push(Command::SetClipTransform {
            clip_id: id.to_owned(),
            scale: Some(after.scale),
            offset_x: Some(after.offset_x),
            offset_y: Some(after.offset_y),
            rotation: Some(after.rotation),
            stretch_x: Some(after.stretch_x),
            stretch_y: Some(after.stretch_y),
        });
    }
    if after.cutout != before.cutout {
        commands.push(Command::SetClipCutout {
            clip_id: id.to_owned(),
            cutout: after.cutout.clone(),
        });
    }
    if after.speed != before.speed {
        commands.push(Command::SetClipSpeed {
            clip_id: id.to_owned(),
            speed: after.speed,
        });
    }
    let mut patch = ClipPatch::default();
    if after.name != before.name {
        patch.name = Some(after.name.clone());
    }
    if after.volume != before.volume {
        patch.volume = Some(after.volume);
    }
    if after.fade_in != before.fade_in {
        patch.fade_in = Some(after.fade_in);
    }
    if after.fade_out != before.fade_out {
        patch.fade_out = Some(after.fade_out);
    }
    if after.opacity != before.opacity {
        patch.opacity = Some(after.opacity);
    }
    if after.preserve_pitch != before.preserve_pitch {
        patch.preserve_pitch = Some(after.preserve_pitch);
    }
    if after.audio_stream != before.audio_stream {
        patch.audio_stream = Some(after.audio_stream);
    }
    if after.flip_h != before.flip_h {
        patch.flip_h = Some(after.flip_h);
    }
    if after.flip_v != before.flip_v {
        patch.flip_v = Some(after.flip_v);
    }
    if after.blend != before.blend {
        patch.blend = Some(if after.blend.is_empty() {
            "normal".to_owned()
        } else {
            after.blend.clone()
        });
    }
    if after.crop != before.crop {
        patch.crop = Some(after.crop);
    }
    commands.extend(key_commands(id, before, after));
    if after.text != before.text {
        patch.text = Some(after.text.clone());
    }
    if after.video_effects != before.video_effects {
        patch.video_effects = Some(after.video_effects.clone());
    }
    if after.filters != before.filters {
        patch.filters = Some(after.filters.clone());
    }
    if patch != ClipPatch::default() {
        commands.push(Command::UpdateClip {
            clip_id: id.to_owned(),
            patch,
        });
    }
    commands
}

/// The commands that turn `before`'s keys into `after`'s: a `SetClipKey`
/// for every key added or changed, a `ClearClipKey` for every key gone.
/// Two keys within a hair of each other are the same key.
fn key_commands(clip_id: &str, before: &Clip, after: &Clip) -> Vec<Command> {
    let near = |a: f64, b: f64| (a - b).abs() <= model::KEY_EPSILON;
    let mut commands = Vec::new();
    for property in model::KeyProperty::ALL {
        for key in after.keys_on(property) {
            let unchanged = before
                .keys_on(property)
                .any(|old| near(old.at, key.at) && old.value == key.value && old.ease == key.ease);
            if !unchanged {
                commands.push(Command::SetClipKey {
                    clip_id: clip_id.to_owned(),
                    property,
                    at: key.at,
                    value: key.value,
                    ease: key.ease,
                });
            }
        }
        for old in before.keys_on(property) {
            if !after.keys_on(property).any(|key| near(old.at, key.at)) {
                commands.push(Command::ClearClipKey {
                    clip_id: clip_id.to_owned(),
                    property,
                    at: old.at,
                });
            }
        }
    }
    commands
}

/// The colour panel's rows: the adjust package's parameters, at the values
/// the clip's chain holds or at the defaults when the clip carries none.
/// Every one can be keyed; `at` is the playhead's place in the clip, `0..=1`,
/// or None when it is outside - a keyed knob then shows its ride's value
/// there, and its cluster says whether a key sits under the playhead.
fn adjust_rows(chain: &[AppliedFilter], at: Option<f64>) -> Vec<AppliedParamData> {
    let Some(package) = Catalogue::builtin().get(ADJUST_ID) else {
        return Vec::new();
    };
    let held = chain.iter().find(|entry| entry.id == ADJUST_ID);
    package
        .manifest
        .params
        .iter()
        // The wheels and curves are drawn by widgets of their own.
        .filter(|param| !param.kind.is_compound())
        .map(|param| {
            let keyed = held.is_some_and(|entry| entry.is_keyed(&param.key));
            let (here, prev, next) = match (held, at) {
                (Some(entry), Some(at)) if keyed => {
                    let (prev, next) = entry.keys_around(&param.key, at);
                    (
                        entry.key_at(&param.key, at).is_some(),
                        prev.is_some(),
                        next.is_some(),
                    )
                }
                _ => (false, false, false),
            };
            let value = match (held, at) {
                (Some(entry), Some(at)) => entry.value_at(&param.key, at, param.default),
                (Some(entry), None) => entry
                    .params
                    .get(&param.key)
                    .copied()
                    .unwrap_or(param.default),
                (None, _) => param.default,
            };
            AppliedParamData {
                entry: -1,
                key: param.key.as_str().into(),
                label: i18n::shelf_text("labels", &param.label).into(),
                group: i18n::shelf_text("groups", &param.group).into(),
                unit: param.unit.as_str().into(),
                min: param.min as f32,
                max: param.max as f32,
                step: if param.step > 0.0 {
                    param.step as f32
                } else {
                    ((param.max - param.min) / 200.0) as f32
                },
                default_value: param.default as f32,
                value: value as f32,
                fmt: format_of(&param.unit),
                keyable: true,
                keyed,
                here,
                prev,
                next,
            }
        })
        .collect()
}

/// A colour knob's value as the document holds it: RGBA packed into one
/// number, red in the top byte, as the shader unpacks it.
fn packed(colour: slint::Color) -> f64 {
    let bytes = [colour.red(), colour.green(), colour.blue(), colour.alpha()];
    f64::from(u32::from_be_bytes(bytes))
}

/// The colour a knob's packed number holds.
fn unpacked(value: f64) -> slint::Color {
    let [r, g, b, a] = (value.clamp(0.0, f64::from(u32::MAX)) as u32).to_be_bytes();
    slint::Color::from_argb_u8(a, r, g, b)
}

/// A chain's colour knobs, a row each, holding the document's colour or
/// the default.
fn chain_colours(chain: &[AppliedFilter]) -> Vec<ColourKnobData> {
    let catalogue = Catalogue::builtin();
    chain
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| {
            let package = catalogue
                .packages()
                .find(|package| package.answers_to(&entry.id))?;
            Some((index, entry, package))
        })
        .flat_map(|(index, entry, package)| {
            package
                .manifest
                .params
                .iter()
                .filter(|param| param.kind == ParamType::Color)
                .map(move |param| ColourKnobData {
                    entry: index as i32,
                    key: param.key.as_str().into(),
                    label: i18n::shelf_text("labels", &param.label).into(),
                    value: unpacked(
                        entry
                            .params
                            .get(&param.key)
                            .copied()
                            .unwrap_or(param.default),
                    ),
                })
        })
        .collect()
}

/// A chain as the inspector's stack draws it: one row per link, and one per
/// knob its package declares, holding the document's value or the default.
/// A link no package answers to keeps its row - so it can be removed - and
/// gets no knobs.
fn chain_rows(chain: &[AppliedFilter]) -> (Vec<AppliedEntryData>, Vec<AppliedParamData>) {
    let catalogue = Catalogue::builtin();
    let mut rows = Vec::new();
    let mut knobs = Vec::new();
    for (index, entry) in chain.iter().enumerate() {
        let package = catalogue
            .packages()
            .find(|package| package.answers_to(&entry.id));
        rows.push(AppliedEntryData {
            id: entry.id.as_str().into(),
            name: package
                .map(|package| package.manifest.effect.name.clone())
                .unwrap_or_else(|| label_of(&entry.id))
                .into(),
            on: entry.enabled,
            known: package.is_some(),
        });
        let Some(package) = package else { continue };
        // The adjust link shows as a link - it can be bypassed or removed
        // here - but its knobs are the Adjust tab's, not the chain's.
        if package.id() == ADJUST_ID {
            continue;
        }
        // Every filter has an intensity whether or not it says so: the one
        // slider a look is expected to have. First, above the look's own.
        if package.kind() == PackageKind::Filter {
            knobs.push(AppliedParamData {
                entry: index as i32,
                key: concat_effects::catalogue::INTENSITY.into(),
                label: t("studio.intensity").into(),
                group: "".into(),
                unit: "%".into(),
                min: 0.0,
                max: 100.0,
                step: 1.0,
                default_value: 100.0,
                value: entry
                    .params
                    .get(concat_effects::catalogue::INTENSITY)
                    .copied()
                    .unwrap_or(100.0) as f32,
                fmt: ParamFormat::Percent,
                keyable: false,
                keyed: false,
                here: false,
                prev: false,
                next: false,
            });
        }
        // A colour is a swatch of its own (see `chain_colours`), and a wheel
        // or a curve is drawn apart (see `chain_grading`).
        for param in package
            .manifest
            .params
            .iter()
            .filter(|param| !param.kind.is_compound() && param.kind != ParamType::Color)
        {
            let step = if param.step > 0.0 {
                param.step
            } else {
                (param.max - param.min) / 200.0
            };
            knobs.push(AppliedParamData {
                entry: index as i32,
                key: param.key.as_str().into(),
                label: i18n::shelf_text("labels", &param.label).into(),
                group: "".into(),
                unit: param.unit.as_str().into(),
                min: param.min as f32,
                max: param.max as f32,
                step: step as f32,
                default_value: param.default as f32,
                value: entry
                    .params
                    .get(&param.key)
                    .copied()
                    .unwrap_or(param.default) as f32,
                fmt: format_of(&param.unit),
                keyable: false,
                keyed: false,
                here: false,
                prev: false,
                next: false,
            });
        }
    }
    (rows, knobs)
}

/// A link's wheels and curves as the inspector draws them: the values the
/// link holds, or the defaults when there is no link yet. `entry` is its
/// index in the chain, -1 for the Adjust panel; `keys` is the playhead's
/// place in the clip when the wheels can carry keys - Some(None) when it is
/// outside - and None when they cannot.
fn grading_rows(
    package: &concat_effects::Package,
    link: Option<&AppliedFilter>,
    entry: i32,
    keys: Option<Option<f64>>,
) -> (Vec<WheelData>, Vec<(CurveData, Vec<CurvePointData>)>) {
    let empty = std::collections::BTreeMap::new();
    let params = link.map_or(&empty, |link| &link.params);
    let mut wheels = Vec::new();
    let mut curves = Vec::new();
    for param in &package.manifest.params {
        match param.kind {
            ParamType::Wheel => {
                let [x, y, m] = grading::wheel_keys(&param.key);
                let at = keys.flatten();
                let value = |name: &str, fallback: f64| match (link, at) {
                    (Some(link), Some(at)) => link.value_at(name, at, fallback),
                    _ => params.get(name).copied().unwrap_or(fallback),
                };
                let keyed = link.is_some_and(|link| link.is_keyed(&m));
                let (here, prev, next) = match (link, at) {
                    (Some(link), Some(at)) if keyed => {
                        let (prev, next) = link.keys_around(&m, at);
                        (
                            link.key_at(&m, at).is_some(),
                            prev.is_some(),
                            next.is_some(),
                        )
                    }
                    _ => (false, false, false),
                };
                wheels.push(WheelData {
                    entry,
                    key: param.key.as_str().into(),
                    label: i18n::shelf_text("labels", &param.label).into(),
                    group: i18n::shelf_text("groups", &param.group).into(),
                    x: value(&x, 0.0) as f32,
                    y: value(&y, 0.0) as f32,
                    master: value(&m, param.default) as f32,
                    min: param.min as f32,
                    max: param.max as f32,
                    default_value: param.default as f32,
                    keyable: keys.is_some(),
                    keyed,
                    here,
                    prev,
                    next,
                });
            }
            ParamType::Curve => {
                let points: Vec<CurvePointData> = grading::stored_points(params, &param.key)
                    .into_iter()
                    .map(|(index, x, y)| CurvePointData {
                        index: index as i32,
                        x: x as f32,
                        y: y as f32,
                    })
                    .collect();
                curves.push((
                    CurveData {
                        entry,
                        key: param.key.as_str().into(),
                        label: i18n::shelf_text("labels", &param.label).into(),
                        group: i18n::shelf_text("groups", &param.group).into(),
                        points: ModelRc::default(),
                        path: grading::curve_path(params, &param.key).into(),
                    },
                    points,
                ));
            }
            _ => {}
        }
    }
    (wheels, curves)
}

/// Every link's wheels and curves in a chain, for the stack.
fn chain_grading(
    chain: &[AppliedFilter],
) -> (Vec<WheelData>, Vec<(CurveData, Vec<CurvePointData>)>) {
    let catalogue = Catalogue::builtin();
    let mut wheels = Vec::new();
    let mut curves = Vec::new();
    for (index, link) in chain.iter().enumerate() {
        let Some(package) = catalogue
            .packages()
            .find(|package| package.answers_to(&link.id))
        else {
            continue;
        };
        if package.id() == ADJUST_ID {
            continue;
        }
        let (more_wheels, more_curves) = grading_rows(package, Some(link), index as i32, None);
        wheels.extend(more_wheels);
        curves.extend(more_curves);
    }
    (wheels, curves)
}

/// The document keys an Adjust knob is keyed under, with what each is worth
/// with no key: a wheel's three - named by the wheel or by any of its own -
/// else the knob's own.
fn adjust_key_names(key: &str) -> Vec<(String, f64)> {
    let Some(package) = Catalogue::builtin().get(ADJUST_ID) else {
        return vec![(key.to_owned(), 0.0)];
    };
    match package.manifest.owner(key) {
        Some(param) if param.kind == ParamType::Wheel => {
            let [x, y, m] = grading::wheel_keys(&param.key);
            vec![(x, 0.0), (y, 0.0), (m, param.default)]
        }
        Some(param) => vec![(key.to_owned(), param.default)],
        None => vec![(key.to_owned(), 0.0)],
    }
}

/// The keys that move with an Adjust knob's key in the keyframes pane: for
/// a wheel's master, the puck's two; nothing for any other knob.
fn wheel_partners(param: &str) -> Vec<String> {
    let names = adjust_key_names(param);
    if names.len() == 3 && names[2].0 == param {
        names.into_iter().take(2).map(|(name, _)| name).collect()
    } else {
        Vec::new()
    }
}

/// The commands that take a wheel's puck keys at `from` to `to` with its
/// master's key - their values and eases kept - when `param` is a wheel's
/// master; nothing for any other knob.
fn partners_moved(
    link: &AppliedFilter,
    clip_id: &str,
    entry: usize,
    param: &str,
    from: f64,
    to: f64,
) -> Vec<Command> {
    wheel_partners(param)
        .into_iter()
        .filter_map(|name| {
            let key = link.keys_on(&name)[link.key_at(&name, from)?];
            Some([
                Command::ClearEffectKey {
                    clip_id: clip_id.to_owned(),
                    entry,
                    key: name.clone(),
                    at: from,
                },
                Command::SetEffectKey {
                    clip_id: clip_id.to_owned(),
                    entry,
                    key: name,
                    at: to,
                    value: key.value,
                    ease: key.ease,
                },
            ])
        })
        .flatten()
        .collect()
}

/// The effects a project's clips name that no package answers to - retired
/// ones, or packages not installed here - by name, each once, in the order
/// they first appear.
fn retired_effects(project: &Project) -> Vec<String> {
    let catalogue = Catalogue::builtin();
    let mut names: Vec<String> = Vec::new();
    for timeline in &project.timelines {
        for clip in &timeline.clips {
            for link in &clip.video_effects {
                if catalogue
                    .packages()
                    .any(|package| package.answers_to(&link.id))
                {
                    continue;
                }
                let name = label_of(&link.id);
                if !names.contains(&name) {
                    names.push(name);
                }
            }
        }
    }
    names
}

/// Names for a notice: the first three, and how many more there are.
fn few(names: &[String]) -> String {
    match names.len() {
        0..=3 => names.join(", "),
        count => format!("{} +{}", names[..3].join(", "), count - 3),
    }
}

/// What the confirmation sheet can ask about.
#[derive(Clone, Debug, PartialEq)]
pub enum Confirm {
    /// Deleting a whole track, and every clip on it.
    RemoveTrack { track_id: String },
    /// Exporting over a file that is already at `path`.
    ReplaceExport { path: String },
}

/// One clip on the clipboard: the clip as it was when copied, and the row
/// of the lane it sat on. A lane's id belongs to its timeline, so a paste
/// into another timeline goes by the row instead and lands the clip on
/// the same row there (#290).
#[derive(Clone, Debug)]
pub struct Held {
    pub row: usize,
    pub clip: Clip,
}

/// Where each held clip lands: `(row, start)` per clip in the order held,
/// or `None` for one with no lane to land on. The group keeps its shape,
/// every clip the same distance from the earliest as when copied, from
/// `base` on; its rows are shifted by `shift` and held inside the `lanes`
/// of the timeline it lands in, and a locked row is passed over for the
/// nearest open one.
fn paste_plan(
    held: &[Held],
    base: f64,
    shift: isize,
    lanes: usize,
    locked: &dyn Fn(usize) -> bool,
) -> Vec<Option<(usize, f64)>> {
    let origin = held
        .iter()
        .map(|held| held.clip.start)
        .fold(f64::INFINITY, f64::min);
    let open: Vec<usize> = (0..lanes).filter(|row| !locked(*row)).collect();
    held.iter()
        .map(|held| {
            let top = isize::try_from(lanes.checked_sub(1)?).ok()?;
            let wanted = (isize::try_from(held.row).ok()? + shift).clamp(0, top);
            let wanted = usize::try_from(wanted).ok()?;
            let row = if locked(wanted) {
                *open.iter().min_by_key(|row| row.abs_diff(wanted))?
            } else {
                wanted
            };
            Some((row, base + (held.clip.start - origin)))
        })
        .collect()
}

/// Each rewritten link as "Old → New", the new one by its package's name.
fn upgraded_effects(pairs: &[(String, String)]) -> Vec<String> {
    let catalogue = Catalogue::builtin();
    pairs
        .iter()
        .map(|(old, new)| {
            let new = catalogue
                .get(new)
                .map(|package| {
                    i18n::package_text(package.id(), "name", &package.manifest.effect.name)
                })
                .unwrap_or_else(|| label_of(new));
            format!("{} → {new}", label_of(old))
        })
        .collect()
}

fn label_of(id: &str) -> String {
    // The name after the namespace: `concat.camera-shake` is Camera Shake.
    let id = id.rsplit('.').next().unwrap_or(id);
    id.split(['-', '_'])
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

impl Studio {
    /// A window with nothing open: the launch screen, with the recents list
    /// read off disk.
    pub fn new(host: Host) -> Self {
        let prefs = Preferences::load(&host.dirs);
        // The remembered log level, unless this run's environment named
        // one: CONCAT_LOG is the louder ask.
        if concat_host::logs::environment_level().is_none()
            && let Some(level) = prefs
                .log_level
                .as_deref()
                .and_then(concat_host::logs::level_named)
            && level != log::max_level()
        {
            concat_host::logs::set_level(level);
        }
        // The words first, so everything published from here on is in
        // the remembered language.
        i18n::select(prefs.locale.as_deref().unwrap_or(i18n::ENGLISH), &host.dirs);
        let languages = i18n::languages(&host.dirs);
        let recents = projects::list(&host.dirs.config);
        let text_presets = presets::all(&host.dirs);
        let mut studio = Self {
            prefs,
            library: Default::default(),
            session: None,
            echo: None,
            empty: Project::new(),
            dirty: false,
            autosave: slint::Timer::default(),
            media: crate::panes::media_bin::MediaBin::default(),
            peaks: HashMap::new(),
            strips: HashMap::new(),
            windows: HashMap::new(),
            art_pending: HashSet::new(),
            art_failed: HashSet::new(),
            art_stamp: None,
            proxied: HashSet::new(),
            window_pending: HashSet::new(),
            waves: RefCell::new(HashMap::new()),
            lanes: crate::panes::timeline::TimelinePane::default(),
            selection: Vec::new(),
            playhead: 0.0,
            // ~20px a second: a ten-second cut fits a pane at its default width.
            playing: false,
            transport: slint::Timer::default(),
            hover: None,
            hover_hush: slint::Timer::default(),
            clipboard: Vec::new(),
            monitor: crate::panes::monitor::MonitorPane::default(),
            export: Default::default(),
            settings: crate::panes::settings::SettingsPane::default(),
            relink: crate::panes::relink::RelinkPane::default(),
            confirm: None,
            transition_selected: None,
            open_menu: -1,
            menu_bar_token: 0,
            menu_target: None,
            menu_media: None,
            menu_token: 0,
            activity_burst: None,
            toast: ToastState::default(),
            on_start: true,
            start: crate::panes::start::StartPane::default(),
            recents,
            posters: HashMap::new(),
            posters_pending: HashSet::new(),
            project_name: "Untitled project".into(),
            dock: default_dock(),
            dock_aside: crate::dock::compact_dock(),
            compact: false,
            workspace: (0.0, 0.0),
            divider_press: None,
            gesture: Gesture::None,
            stage_guides: Vec::new(),
            inspector_jump: (0, "", ""),
            media_jump: 0,
            audition: None,
            revision: 0,
            flat: None,
            commit_pending: false,
            commit_targets: Vec::new(),
            commit_timer: slint::Timer::default(),
            shelf_stamp: std::cell::RefCell::new(None),
            look_art: std::cell::RefCell::new(HashMap::new()),
            card_paths: HashMap::new(),
            cards_drawn: 0,
            cards_busy: false,
            cards_again: false,
            packages_watch: slint::Timer::default(),
            packages_seen: 0,
            packages_pending: None,
            last_commit: None,
            title_blocks: HashMap::new(),
            drop: None,
            project_sheet: crate::panes::project::ProjectPane::default(),
            captions: crate::panes::captions::CaptionsPane::default(),
            speech: crate::panes::speech::SpeechPane::default(),
            source: None,
            meters: crate::meters::MeterFeed::default(),
            voiceover: crate::panes::voiceover::VoiceoverPane::default(),
            text_presets,
            installed_fonts: presets::installed_fonts(&host.dirs),
            system_fonts: Vec::new(),
            fonts_published: std::cell::Cell::new(None),
            font_query: String::new(),
            font_search_published: std::cell::Cell::new(None),
            languages,
            brush: 0,
            brush_size: 0.06,
            painting: true,
            cutout_jobs: HashMap::new(),
            analysis_tool: 0,
            analysis_colour: 3,
            football_pending: None,
            dock_before_analysis: None,
            calibrating: None,
            board_cache: RefCell::new(None),
            tag_current: None,
            eval_profile: 0,
            eval_team: 1,
            assistant_lines: Vec::new(),
            assistant_history: Vec::new(),
            assistant_busy: false,
            assistant_provider_used: String::new(),
            assistant_recorder: None,
            library_queue: Vec::new(),
            assistant_hearing: false,
            assistant_config: RefCell::new(None),
            eval_library_cache: RefCell::new(None),
            eval_context_cache: RefCell::new(None),
            tag_filter: None,
            tag_min_rating: 0,
            tag_editing: false,
            draw_tool: None,
            draw_style: DrawStyle::default(),
            draw_draft: None,
            mark_in: None,
            mark_out: None,
            football_picking: false,
            football_chosen: None,
            football_job: None,
            football_tried: HashSet::new(),
            enhance_jobs: HashMap::new(),
            reverse_jobs: HashMap::new(),
            region_job: None,
            pending_stroke: None,
            host,
        };
        studio.handle(crate::panes::Msg::Settings(
            crate::panes::settings::SettingsMsg::Restore,
        ));
        // The effects shelf starts on "All", not whatever a stale default
        // would leave it on.
        studio.library[1].category = "All".to_owned();
        studio.library[1].group = -1;
        studio
    }

    // ── reading the edit ──

    /// The project as the window should draw it: the echo while a gesture
    /// runs, the session's otherwise, and nothing at all on the launch screen.
    pub fn project(&self) -> &Project {
        self.echo
            .as_ref()
            .or_else(|| self.session.as_ref().map(|session| session.project()))
            .unwrap_or(&self.empty)
    }

    pub fn timeline(&self) -> &Timeline {
        self.project().active()
    }

    pub fn clip(&self, id: &str) -> Option<&Clip> {
        self.timeline().clip(id)
    }

    /// The active timeline's frame rate. With no project open, the empty
    /// project's one timeline answers, at the default.
    pub fn frame_rate(&self) -> f32 {
        self.project().active().video.rate() as f32
    }

    /// The active timeline's output size.
    pub fn output_size(&self) -> (u32, u32) {
        let video = self.project().active().video;
        (video.width, video.height)
    }

    /// The monitor's quality tier for the active timeline: 0 full, 1 half,
    /// 2 quarter; see `MonitorPane::quality_of`.
    pub fn quality_of(&self) -> usize {
        self.monitor.quality_of(self.project())
    }

    /// The track a row index names. Rows count from the top of the panel and
    /// the model stores lanes bottom-most first - the compositing order - so
    /// this is the one place the two orders meet.
    /// Asks before the track on lane `row` goes, empty or not: a track is
    /// a whole lane of the cut, and the button that drops it sits beside
    /// the ones that hide and mute it.
    /// Lets go of every picture and waveform made for the project's media,
    /// as a project closes or another opens: the bin's thumbnails, the
    /// filmstrips, the waveform pyramids and their drawn envelopes. They
    /// were never dropped, so each project opened in a session added its
    /// art to the last one's until the app quit.
    fn forget_art(&mut self) {
        self.peaks.clear();
        self.strips.clear();
        self.windows.clear();
        self.art_pending.clear();
        self.window_pending.clear();
        self.waves.borrow_mut().clear();
        self.media.thumbs.clear();
    }

    pub fn ask_remove_track(&mut self, row: i32) {
        let Some(track_id) = self.row_track(row).map(|track| track.id.clone()) else {
            return;
        };
        self.confirm = Some(Confirm::RemoveTrack { track_id });
    }

    /// The confirmation sheet's answer: go ahead with what it asked about.
    pub fn confirm_accepted(&mut self) {
        match self.confirm.take() {
            Some(Confirm::RemoveTrack { track_id }) => {
                self.apply(Command::RemoveTrack { track_id });
            }
            Some(Confirm::ReplaceExport { .. }) => {
                self.handle(crate::panes::Msg::Export(
                    crate::panes::export::ExportMsg::Overwrite,
                ));
            }
            None => {}
        }
    }

    /// The confirmation sheet, as Slint shows it.
    fn confirm_data(&self) -> crate::ui::ConfirmData {
        let Some(confirm) = &self.confirm else {
            return crate::ui::ConfirmData::default();
        };
        match confirm {
            Confirm::RemoveTrack { track_id } => {
                let clips = self
                    .timeline()
                    .clips
                    .iter()
                    .filter(|clip| &clip.track_id == track_id)
                    .count();
                let message = match clips {
                    0 => t("timeline.deleteTrackEmpty"),
                    1 => t("timeline.deleteTrackOneClip"),
                    many => tf("timeline.deleteTrackClips", &[&many]),
                };
                crate::ui::ConfirmData {
                    open: true,
                    title: t("timeline.deleteTrackTitle").into(),
                    message: message.into(),
                    action: t("common.delete").into(),
                }
            }
            Confirm::ReplaceExport { path } => crate::ui::ConfirmData {
                open: true,
                title: t("export.replaceFileTitle").into(),
                message: tf("export.replaceFileMessage", &[path]).into(),
                action: t("export.replace").into(),
            },
        }
    }

    pub fn row_track(&self, row: i32) -> Option<&Track> {
        let tracks = &self.timeline().tracks;
        let count = tracks.len() as i32;
        if row < 0 || row >= count {
            return None;
        }
        tracks.get((count - 1 - row) as usize)
    }

    pub fn row_of(&self, track_id: &str) -> i32 {
        let tracks = &self.timeline().tracks;
        tracks
            .iter()
            .position(|track| track.id == track_id)
            .map(|index| tracks.len() as i32 - 1 - index as i32)
            .unwrap_or(0)
    }

    pub fn locked(&self, track_id: &str) -> bool {
        self.lanes
            .lane_view
            .get(track_id)
            .is_some_and(|view| view.locked)
    }

    fn lane_size(&self, track_id: &str) -> TrackSize {
        self.lanes
            .lane_view
            .get(track_id)
            .map_or(TrackSize::Auto, |view| view.size)
    }

    /// How tall one lane is drawn: a lane takes the height of the tallest
    /// thing on it, without the lanes having to be typed. An empty one takes
    /// the middle size.
    fn lane_height(&self, lane: &Track) -> f32 {
        match self.lane_size(&lane.id) {
            TrackSize::Small => LANE_SMALL,
            TrackSize::Medium => LANE_MEDIUM,
            TrackSize::Large => LANE_LARGE,
            TrackSize::Auto => {
                let tallest = self
                    .timeline()
                    .clips
                    .iter()
                    .filter(|clip| clip.track_id == lane.id)
                    .map(|clip| match clip.kind {
                        model::ClipKind::Video | model::ClipKind::Image => LANE_LARGE,
                        model::ClipKind::Audio => LANE_MEDIUM,
                        // A title is its name strip alone; a layer has no
                        // picture at all; a shape's name says what it is.
                        // None needs a body's height.
                        model::ClipKind::Text | model::ClipKind::Layer | model::ClipKind::Shape => {
                            LANE_SMALL
                        }
                    })
                    .fold(0.0_f32, f32::max);
                if tallest > 0.0 { tallest } else { LANE_MEDIUM }
            }
        }
    }

    /// Every lane's height, top-most first.
    pub fn lane_heights(&self) -> Vec<f32> {
        self.timeline()
            .tracks
            .iter()
            .rev()
            .map(|lane| self.lane_height(lane))
            .collect()
    }

    pub fn row_at(&self, y: f32) -> i32 {
        row_at(&self.lane_heights(), y)
    }

    /// Seconds the project runs to, for Fit and for the scroll floor.
    pub fn duration(&self) -> f32 {
        self.timeline().clips.iter().fold(0.0_f64, |longest, clip| {
            longest.max(clip.start + clip.duration)
        }) as f32
    }

    /// Clip edges, the playhead and zero all pull, and the nearest inside
    /// the threshold wins.
    pub fn snapped(&self, time: f32, threshold: f32, exclude: &str) -> f32 {
        if !self.lanes.snap {
            return time;
        }
        let mut best = time;
        let mut best_distance = threshold;
        let mut consider = |target: f32| {
            let distance = (target - time).abs();
            if distance < best_distance {
                best = target;
                best_distance = distance;
            }
        };
        consider(0.0);
        consider(self.playhead);
        for clip in &self.timeline().clips {
            if clip.id == exclude {
                continue;
            }
            consider(clip.start as f32);
            consider((clip.start + clip.duration) as f32);
        }
        best
    }

    // ── changing the edit ──

    /// Applies one command to the session and reports the id it minted.
    /// A refusal becomes a notice; the echo, if any, is dropped either way,
    /// because the session's project is the truth again.
    /// Logs one line of the activity trail, folding a burst of one kind
    /// of message into its first line and, when the burst ends - a
    /// different message, an edit, or a pause - one line with the count.
    fn note_activity(&mut self, activity: crate::panes::Activity) {
        use crate::panes::Burst;
        let now = std::time::Instant::now();
        let Some(key) = activity.burst else {
            self.end_burst();
            log::log!(activity.level, "{}", activity.line);
            return;
        };
        match &mut self.activity_burst {
            Some(burst) if burst.key == key && now.duration_since(burst.last) < Burst::GAP => {
                burst.count += 1;
                burst.line = activity.line;
                burst.last = now;
            }
            _ => {
                self.end_burst();
                log::log!(activity.level, "{}", activity.line);
                self.activity_burst = Some(Burst {
                    key,
                    line: activity.line,
                    count: 1,
                    began: now,
                    last: now,
                });
            }
        }
    }

    /// Ends the burst being folded, if one is, with its count.
    fn end_burst(&mut self) {
        if let Some(burst) = self.activity_burst.take() {
            burst.flush();
        }
    }

    pub fn apply(&mut self, command: Command) -> Option<String> {
        // An edit ends whatever burst the view was in.
        self.end_burst();
        log::info!("edit: {}", command_name(&command));
        self.flush_commit();
        self.echo = None;
        // Anything but an inspector commit ends the coalescing window; the
        // commit path sets `last_commit` again right after calling here.
        self.last_commit = None;
        let session = self.session.as_mut()?;
        let chosen = matches!(command, Command::SetTimelineVideo { .. });
        let before = session.video().color_space;
        match session.apply(command) {
            Ok(view) => {
                let after = session.video().color_space;
                self.after_change();
                if !chosen {
                    self.tell_of_hdr(before, after);
                }
                view.created_id
            }
            Err(error) => {
                self.notify(&error, true);
                None
            }
        }
    }

    /// [`Studio::apply`] as one move of an inspector gesture: the same
    /// bookkeeping, but the editor folds it into the gesture's undo step
    /// and the coalescing window stays open for the next move.
    fn apply_within(&mut self, gesture: &str, command: Command) -> Option<String> {
        // A knob's gesture, folded into one step; said at Debug, since a
        // dial turned is many of these.
        log::debug!("edit: {} ({gesture})", command_name(&command));
        self.echo = None;
        let session = self.session.as_mut()?;
        let before = session.video().color_space;
        match session.apply_within(Some(gesture), command) {
            Ok(view) => {
                let after = session.video().color_space;
                self.after_change();
                self.tell_of_hdr(before, after);
                view.created_id
            }
            Err(error) => {
                self.notify(&error, true);
                None
            }
        }
    }

    /// [`Studio::apply`], or [`Studio::apply_within`] when `gesture` names
    /// one: for the edits that are a step of their own from one caller and
    /// part of a larger step from another, as laying a copy is.
    fn apply_in(&mut self, gesture: Option<&str>, command: Command) -> Option<String> {
        match gesture {
            Some(gesture) => self.apply_within(gesture, command),
            None => self.apply(command),
        }
    }

    /// Says so when an edit turned the active timeline HDR - its first HDR
    /// clip, which the editor follows (see `Editor::follow_first_hdr`) - so
    /// the change is never a surprise, and says where it is set back.
    fn tell_of_hdr(&mut self, before: model::ColorSpace, after: model::ColorSpace) {
        if before.is_sdr() && after.is_hdr() {
            let name = self.timeline().name.clone();
            self.notify(&tf("studio.timelineNowHdr", &[&name]), false);
        }
    }

    /// The bookkeeping every change to the edit needs: the caches that
    /// follow the document, the autosave, the monitor and the mix.
    fn after_change(&mut self) {
        self.dirty = true;
        self.revision += 1;
        self.assign_media_rows();
        let survivors: HashSet<String> = self
            .timeline()
            .clips
            .iter()
            .map(|clip| clip.id.clone())
            .collect();
        self.selection.retain(|id| survivors.contains(id));
        self.schedule_autosave();
        self.sync_audio();
        self.request_media_art();
        self.request_preview();
        self.ensure_cutouts();
        self.ensure_football_silhouettes();
        self.ensure_regions();
    }

    pub fn undo(&mut self) {
        self.flush_commit();
        self.echo = None;
        if let Some(session) = self.session.as_mut()
            && session.can_undo()
        {
            session.undo();
            self.after_change();
        }
    }

    pub fn redo(&mut self) {
        self.flush_commit();
        self.echo = None;
        if let Some(session) = self.session.as_mut()
            && session.can_redo()
        {
            session.redo();
            self.after_change();
        }
    }

    /// Starts a gesture's preview: the echo is a clone of the project the
    /// pointer will mutate.
    pub fn begin_echo(&mut self) {
        if self.echo.is_none() {
            self.echo = Some(self.project().clone());
        }
    }

    pub fn echo_clip_mut(&mut self, id: &str) -> Option<&mut Clip> {
        self.echo.as_mut()?.active_mut().clip_mut(id)
    }

    fn schedule_autosave(&mut self) {
        // A second and a half after the last change, not after the first:
        // a drag that commits ten edits saves once.
        self.autosave.start(
            slint::TimerMode::SingleShot,
            std::time::Duration::from_millis(1500),
            || {
                crate::host::Shell::with(|shell, app| {
                    shell.studio.borrow_mut().save(false);
                    shell.studio.borrow().publish(&app, &shell.models);
                });
            },
        );
    }

    /// Writes the document, on a worker. `announce` says so on success;
    /// a failure is always said.
    pub fn save(&mut self, announce: bool) {
        let Some(session) = self.session.as_mut() else {
            return;
        };
        self.autosave.stop();
        let (path, document) = session.prepare_save(None);
        self.dirty = false;
        spawn(
            move || projects::save(&path, &document),
            move |studio, _, _, result| match result {
                Ok(()) if announce => studio.notify(&t("studio.projectSaved"), false),
                Ok(()) => {}
                Err(error) => {
                    studio.dirty = true;
                    studio.notify(&tf("studio.couldNotSave", &[&error]), true);
                }
            },
        );
    }

    /// The audible clip set, handed to playback whenever the edit changes.
    /// Not while a library file has the speakers: the timeline's sound is
    /// handed back when it lets them go.
    pub(crate) fn sync_audio(&mut self) {
        if self.source.is_some() {
            return;
        }
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let specs = audio_specs(&session.flattened_clips());
        self.host
            .playback
            .set_clips(std::path::PathBuf::from(session.path()), specs);
    }

    // ── the monitor ──

    /// Asks the monitor for the frame at the playhead; see
    /// `MonitorPane`.
    pub fn request_preview(&mut self) {
        self.handle(crate::panes::Msg::Monitor(
            crate::panes::monitor::MonitorMsg::Request,
        ));
    }

    /// What the monitor draws at the playhead: the document flattened with
    /// its titles, plus the frame's own additions - a look being shown, a
    /// cutout being painted - and the session's settings. None without a
    /// project.
    pub fn preview_clips(
        &mut self,
    ) -> Option<(
        std::sync::Arc<Vec<concat_export::ExportClip>>,
        concat_project::DocumentSettings,
    )> {
        let session = self.session.as_ref()?;
        // A library file on the monitor: its clip alone, flattened once.
        if let Some(source) = self.source.as_ref() {
            return Some((std::sync::Arc::clone(&source.clips), session.settings()));
        }
        // The echo when there is one: a picture being dragged on the stage
        // is drawn where the pointer has it, not where the document last
        // had it. Same flattening the session does for itself, project
        // folder included: that is what names a cutout's masks, and
        // without it the monitor would show every cutout as shot.
        let project_dir = std::path::PathBuf::from(session.path());
        let (width, height) = self.output_size();
        // The flattening is the document's, not the frame's: kept between
        // frames of an unchanged document and handed over by pointer, so
        // playback and scrubbing neither flatten nor plan again. A gesture
        // in flight - the echo - is not the document, and flattens fresh.
        let cached = self.echo.is_none()
            && self
                .flat
                .as_ref()
                .is_some_and(|(at, w, h, _)| *at == self.revision && *w == width && *h == height);
        let clips = if cached {
            std::sync::Arc::clone(&self.flat.as_ref().expect("checked").3)
        } else {
            let mut clips = concat_export::flatten::flatten_timeline_in(
                self.project(),
                None,
                Some(&project_dir),
            );
            // Titles, painted to pictures and rejoined; see concat-host's titles.
            if self.echo.is_some() {
                // A change in flight: the titles are painted in memory at
                // the monitor's size and handed to it as pixels - no file
                // per pointer step - and the blocks they report are scaled
                // back up to the output's terms, which the stage measures
                // in. Nothing is written until the change is committed.
                let (shown_w, shown_h) = crate::panes::monitor::MonitorPane::frame_size(
                    self.quality_of(),
                    (width, height),
                );
                let scale = f64::from(shown_w) / f64::from(width);
                let up = |px: u32| (f64::from(px) / scale).round() as u32;
                let up_off = |px: i32| (f64::from(px) / scale).round() as i32;
                for title in self
                    .host
                    .titles
                    .clips_live(self.project(), shown_w, shown_h)
                {
                    self.title_blocks.insert(
                        title.clip_id,
                        (
                            (up(title.block.0), up(title.block.1)),
                            (up_off(title.offset.0), up_off(title.offset.1)),
                        ),
                    );
                    if let Some(frame) = title.frame {
                        self.host
                            .monitor
                            .hold_still(std::path::Path::new(&title.clip.path), frame);
                    }
                    clips.push(title.clip);
                }
            } else {
                for title in self.host.titles.clips(self.project(), width, height) {
                    self.title_blocks
                        .insert(title.clip_id, (title.block, title.offset));
                    clips.push(title.clip);
                }
            }
            let clips = std::sync::Arc::new(clips);
            if self.echo.is_none() {
                self.flat = Some((self.revision, width, height, std::sync::Arc::clone(&clips)));
            }
            clips
        };
        // The frame's own additions - a look being shown, a cutout being
        // painted - go on a copy, so the kept list stays the document's.
        let mut own: Option<Vec<concat_export::ExportClip>> = None;
        // The look being shown before it is laid down goes into this
        // frame only, as the layer it would be: over every track, the
        // whole way along, at full strength. The timeline is as it was.
        if let Some(filter_id) = self.audition.clone() {
            let effects = vec![AppliedFilter::new(filter_id)];
            let track = clips
                .iter()
                .map(|flat| flat.track)
                .max()
                .map_or(0, |top| top + 1);
            own.get_or_insert_with(|| (*clips).clone())
                .push(concat_export::ExportClip {
                    muted: true,
                    volume: 0.0,
                    effects,
                    has_audio: Some(false),
                    ..concat_export::ExportClip::blank(
                        concat_export::ClipKind::Layer,
                        0.0,
                        f64::from(self.duration()).max(1.0),
                        track,
                    )
                });
        }
        // While the brushes are out, the clip being painted is drawn with
        // its cutout tinted over the whole picture rather than cut, so a
        // stroke shows what it grabbed against what it left.
        if self.painting
            && let Some(target) = self.paint_target()
            && let Some(media) = self.project().media_by_id(&target.media_id)
        {
            let (path, start) = (media.path.clone(), target.start);
            if let Some(flat) = own
                .get_or_insert_with(|| (*clips).clone())
                .iter_mut()
                .find(|flat| flat.path == path && (flat.start - start).abs() < 1e-6)
            {
                flat.highlighted = true;
            }
        }
        // While a player is being picked, the selected clip outlines every
        // tracked box, the chosen one in the accent.
        if self.football_picking
            && let Some(target) = self.sole_selection().and_then(|id| self.clip(&id).cloned())
            && let Some(media) = self.project().media_by_id(&target.media_id)
        {
            let (path, start) = (media.path.clone(), target.start);
            if let Some(flat) = own
                .get_or_insert_with(|| (*clips).clone())
                .iter_mut()
                .find(|flat| flat.path == path && (flat.start - start).abs() < 1e-6)
            {
                flat.football_picking = true;
                flat.football_chosen = self.football_chosen;
            }
        }
        let clips = own.map(std::sync::Arc::new).unwrap_or(clips);
        Some((clips, session.settings()))
    }

    // ── playback ──

    pub fn play_toggle(&mut self) {
        // The monitor's play button and Space play what it shows.
        if self.source.is_some() {
            self.source_toggle();
            return;
        }
        if self.playing {
            self.pause();
            return;
        }
        if self.session.is_none() {
            return;
        }
        // Playing from the tail would sit there doing nothing, so the
        // button rewinds first - what every editor does.
        if self.playhead >= self.duration() {
            self.playhead = 0.0;
        }
        // The transport owns the monitor from here; a frame from under a
        // pointer that happens to be over the lanes must not stay on it.
        self.end_hover();
        self.playing = true;
        log::debug!("playback: playing from {:.3}s", self.playhead);
        self.host.playback.play(f64::from(self.playhead));
        self.meters.wake();
        // The clock is the audio device's. This looks at it far more often
        // than any timeline's frame rate and acts only when the frame under
        // it changes, so each frame is asked for within a few milliseconds of
        // its time instead of whenever a fixed 30 Hz beat next comes round -
        // which at 24, 25 or 60 fps is unevenly late, and reads as judder.
        let mut shown: Option<i64> = None;
        self.transport.start(
            slint::TimerMode::Repeated,
            std::time::Duration::from_millis(8),
            move || {
                crate::host::Shell::with(|shell, app| {
                    {
                        let mut studio = shell.studio.borrow_mut();
                        let end = studio.duration();
                        let position = studio.host.playback.position_now() as f32;
                        let fps = studio.project().active().video.rate();
                        let frame = (f64::from(position) * fps + 1e-6).floor() as i64;
                        if position < end && shown == Some(frame) {
                            return;
                        }
                        shown = Some(frame);
                        studio.playhead = position.min(end);
                        // The view follows: a playhead that runs off the
                        // right edge, or sits off the left, pages the lanes
                        // to it, the way every editor keeps the cut in view.
                        // A phone's lanes glide instead, the playhead held
                        // at their middle.
                        if studio.lanes.centred {
                            let playhead = studio.playhead;
                            studio.lanes.recentre(playhead);
                        } else if let Some((low, high)) = studio.lanes.published_span() {
                            let screen = (high - low) / 3.0;
                            let left = studio.lanes.scroll_left;
                            if screen > 0.0
                                && (studio.playhead > left + screen * 0.95
                                    || studio.playhead < left)
                            {
                                studio.lanes.scroll_left =
                                    (studio.playhead - screen * 0.05).max(0.0);
                            }
                        }
                        if position >= end {
                            studio.pause();
                        } else {
                            studio.request_preview();
                        }
                    }
                    shell.studio.borrow().publish_lanes(&app, &shell.models);
                });
            },
        );
    }

    pub fn pause(&mut self) {
        if self.playing {
            log::debug!("playback: paused at {:.3}s", self.playhead);
        }
        let was_playing = self.playing;
        self.playing = false;
        self.transport.stop();
        self.host.playback.pause();
        // The textures playback streamed through are let go once it stops.
        if was_playing {
            self.host.monitor.shrink();
        }
    }

    /// Moves the playhead, the transport with it, and asks for the frame.
    /// Never before zero; past the end of the content only when Settings
    /// lets it, which is the default, so the ruler can be clicked beyond the
    /// last clip and something placed at the playhead there.
    pub fn seek(&mut self, seconds: f32) {
        // The timeline was clicked: it has the monitor back.
        self.close_source();
        self.playhead = seconds.max(0.0);
        if self.prefs.playhead_stops_at_end {
            self.playhead = self.playhead.min(self.duration().max(0.0));
        }
        // A phone's lanes move under a playhead held at their middle.
        self.lanes.recentre(self.playhead);
        self.host.playback.seek(f64::from(self.playhead));
        self.request_preview();
    }

    // ── the preview axis ──

    /// The instant the monitor shows: the pointer's while it crosses the
    /// lanes with the preview axis on, else the playhead's.
    pub fn preview_time(&self) -> f32 {
        if let Some(source) = self.source.as_ref() {
            return source.time as f32;
        }
        self.hover.unwrap_or(self.playhead)
    }

    /// The pointer is over the lanes at `seconds`. Nothing while playing -
    /// the transport owns the monitor then - and nothing with the axis off.
    pub fn hover(&mut self, seconds: f32) {
        if !self.prefs.preview_axis || self.playing {
            return;
        }
        let seconds = seconds.max(0.0);
        if self.hover == Some(seconds) {
            return;
        }
        self.hover = Some(seconds);
        if self.prefs.preview_axis_audio {
            // A burst of the sound under the pointer: the engine plays from
            // there and is stopped a beat later, then put back where the
            // playhead is, so the next Play starts where the cut says.
            self.host.playback.play(f64::from(seconds));
            let playhead = f64::from(self.playhead);
            self.hover_hush.start(
                slint::TimerMode::SingleShot,
                std::time::Duration::from_millis(120),
                move || {
                    crate::host::Shell::with(|shell, _| {
                        let studio = shell.studio.borrow();
                        if !studio.playing {
                            studio.host.playback.pause();
                            studio.host.playback.seek(playhead);
                        }
                    });
                },
            );
        }
        self.request_preview();
    }

    /// The pointer left the lanes, or the axis went off: the monitor is the
    /// playhead's again.
    pub fn end_hover(&mut self) {
        if self.hover.take().is_some() {
            self.request_preview();
        }
    }

    // ── the bin ──

    /// Gives every media item the row Slint knows it by; see
    /// `MediaBin::assign_rows`.
    fn assign_media_rows(&mut self) {
        let mut bin = std::mem::take(&mut self.media);
        bin.assign_rows(self.project());
        self.media = bin;
    }

    /// Decodes art for every media item that has none yet: its pictures
    /// and the waveform of its default audio stream - and, for every clip
    /// that plays another of its media's streams, that stream's waveform
    /// too, so a lane shows the sound it will make.
    pub(crate) fn request_media_art(&mut self) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let project_path = session.path().to_owned();
        // A file larger than HD gets a proxy for playback and the
        // filmstrips, written once on the scheduler's proxy lane; see
        // concat_host::proxy. Asked once per file per project.
        type Unproxied = (
            String,
            String,
            u32,
            u32,
            Option<concat_media::ColorRange>,
            bool,
        );
        let unproxied: Vec<Unproxied> = self
            .project()
            .media
            .iter()
            .filter(|item| item.kind == model::MediaKind::Video && !item.placeholder)
            .filter(|item| !self.proxied.contains(&item.id))
            .filter_map(|item| {
                let (width, height) = (item.width?, item.height?);
                Some((
                    item.id.clone(),
                    item.path.clone(),
                    width,
                    height,
                    item.color_range.map(concat_export::engine_range),
                    item.color_space.is_hdr(),
                ))
            })
            .collect();
        for (id, path, width, height, range, hdr) in unproxied {
            concat_host::proxy::ensure(
                std::path::Path::new(&project_path),
                &path,
                width,
                height,
                range,
                hdr,
            );
            self.proxied.insert(id);
        }
        /// One job: the art key it fills, and what to decode.
        struct Want {
            key: String,
            id: String,
            path: String,
            kind: model::MediaKind,
            has_audio: bool,
            duration: Option<f64>,
            stream: Option<u32>,
            pictures: bool,
            /// The levels the file is read as, so the filmstrip reads the
            /// proxy written for that reading.
            range: Option<concat_media::ColorRange>,
        }
        let project = self.project();
        let mut wanted: Vec<Want> = project
            .media
            .iter()
            .filter(|item| !item.placeholder && !item.path.is_empty())
            .filter(|item| {
                !self.art_pending.contains(&item.id) && !self.art_failed.contains(&item.id)
            })
            .filter(|item| {
                let needs_thumb = item.kind != model::MediaKind::Audio
                    && (!self.media.thumbs.contains_key(&item.id)
                        || !self.strips.contains_key(&item.id));
                let needs_peaks = (item.kind == model::MediaKind::Audio || item.has_audio)
                    && !self.peaks.contains_key(&item.id);
                needs_thumb || needs_peaks
            })
            .map(|item| Want {
                key: item.id.clone(),
                id: item.id.clone(),
                path: item.path.clone(),
                kind: item.kind,
                has_audio: item.has_audio,
                duration: item.duration,
                stream: None,
                pictures: item.kind != model::MediaKind::Audio,
                range: item.color_range.map(concat_export::engine_range),
            })
            .collect();
        // The other streams clips have chosen, once each.
        let mut asked: HashSet<String> = wanted.iter().map(|want| want.key.clone()).collect();
        for clip in project
            .timelines
            .iter()
            .flat_map(|timeline| timeline.clips.iter())
        {
            let Some(stream) = clip.audio_stream else {
                continue;
            };
            let Some(item) = project.media_by_id(&clip.media_id) else {
                continue;
            };
            if item.placeholder
                || item.path.is_empty()
                || !(item.kind == model::MediaKind::Audio || item.has_audio)
            {
                continue;
            }
            let key = art_key(&item.id, Some(stream));
            if self.peaks.contains_key(&key)
                || self.art_pending.contains(&key)
                || self.art_failed.contains(&key)
                || !asked.insert(key.clone())
            {
                continue;
            }
            wanted.push(Want {
                key,
                id: item.id.clone(),
                path: item.path.clone(),
                kind: item.kind,
                has_audio: item.has_audio,
                duration: item.duration,
                stream: Some(stream),
                pictures: false,
                range: item.color_range.map(concat_export::engine_range),
            });
        }
        for want in wanted {
            let Want {
                key,
                id,
                path,
                kind,
                has_audio,
                duration,
                stream,
                mut pictures,
                range,
            } = want;

            // Restore pictures from the project's JPEG artwork cache before
            // starting a decoder. Peaks already have their own disk cache in
            // concat-media; thumbnails and filmstrips are what used to be
            // recreated on every project open.
            if stream.is_none() && kind != model::MediaKind::Audio {
                let cached = cached_media_art(&project_path, &id, &path, kind);
                if let Some(image) = cached.thumbnail {
                    self.media.thumbs.insert(id.clone(), image);
                }
                if let Some(strip) = cached.strip {
                    self.strips.insert(id.clone(), strip.into());
                }
                pictures = !self.media.thumbs.contains_key(&id) || !self.strips.contains_key(&id);
            }

            let needs_peaks =
                (kind == model::MediaKind::Audio || has_audio) && !self.peaks.contains_key(&key);
            if !pictures && !needs_peaks {
                continue;
            }

            self.art_pending.insert(key);
            let project = project_path.clone();
            // On the artwork lane, a few at a time - see host.rs.
            spawn_art(
                move || {
                    media_art(
                        id, path, kind, has_audio, duration, project, stream, pictures, range,
                    )
                },
                |studio, _, _, art: MediaArt| {
                    let key = art_key(&art.id, art.stream);
                    studio.art_pending.remove(&key);
                    if art.thumbnail.is_none() && art.strip.is_none() && art.peaks.is_none() {
                        studio.art_failed.insert(key.clone());
                    }
                    if let Some(frame) = art.thumbnail {
                        studio.media.thumbs.insert(art.id.clone(), image_of(&frame));
                    }
                    if let Some((frame, frames)) = art.strip {
                        studio
                            .strips
                            .insert(art.id.clone(), Strip::of(&frame, frames));
                    }
                    if let Some(peaks) = art.peaks {
                        let prefix = format!("{key}|");
                        studio.peaks.insert(key, peaks);
                        studio
                            .waves
                            .borrow_mut()
                            .retain(|key, _| !key.starts_with(&prefix));
                    }
                },
            );
        }
        self.request_window_art();
    }

    /// Decodes a cell strip for every picture clip on the current timeline
    /// whose cut is too short a piece of its footage for the file's own
    /// strip - see `host::strip_window` - and lets go of the cells nothing
    /// shows once there are more than `WINDOWS_KEPT` of them.
    fn request_window_art(&mut self) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let project_path = session.path().to_owned();
        struct Want {
            key: String,
            id: String,
            path: String,
            level: u32,
            cell: u32,
            duration: f64,
        }
        let mut shown: HashSet<String> = HashSet::new();
        let mut wanted: Vec<Want> = Vec::new();
        for clip in &self.timeline().clips {
            if clip.kind != model::ClipKind::Video {
                continue;
            }
            let Some(item) = self.project().media_by_id(&clip.media_id) else {
                continue;
            };
            if item.placeholder || item.path.is_empty() {
                continue;
            }
            let Some((start, span, duration)) = self.cut_of(clip) else {
                continue;
            };
            let Some((level, cell)) = strip_window(start, span, duration) else {
                continue;
            };
            let key = window_key(&clip.media_id, level, cell);
            if !shown.insert(key.clone())
                || self.windows.contains_key(&key)
                || self.window_pending.contains(&key)
            {
                continue;
            }
            wanted.push(Want {
                key,
                id: item.id.clone(),
                path: item.path.clone(),
                level,
                cell,
                duration,
            });
        }
        if self.windows.len() > WINDOWS_KEPT {
            self.windows.retain(|key, _| shown.contains(key));
        }
        for want in wanted {
            let Want {
                key,
                id,
                path,
                level,
                cell,
                duration,
            } = want;
            if let Some(strip) = cached_window_art(&project_path, &id, &path, level, cell) {
                self.windows.insert(key, strip.into());
                continue;
            }
            self.window_pending.insert(key);
            let project = project_path.clone();
            spawn_strip(
                move || window_art(id, path, project, level, cell, duration),
                |studio, _, _, art: WindowArt| {
                    let key = window_key(&art.id, art.level, art.cell);
                    studio.window_pending.remove(&key);
                    if let Some((frame, frames)) = art.strip {
                        studio.windows.insert(key, Strip::of(&frame, frames));
                    }
                },
            );
        }
    }

    /// A picture clip's cut as fractions of its footage - where it begins
    /// and how much it covers - with the footage's length in seconds.
    /// `None` for a still or footage of unknown length: one frame, all of it.
    fn cut_of(&self, clip: &Clip) -> Option<(f64, f64, f64)> {
        let seconds = self
            .project()
            .media
            .iter()
            .find(|item| item.id == clip.media_id)
            .and_then(|item| item.duration)
            .filter(|seconds| *seconds > 0.0)?;
        Some((
            (clip.source_start / seconds).clamp(0.0, 1.0),
            (clip.duration * clip.speed / seconds).clamp(0.0, 1.0),
            seconds,
        ))
    }

    /// The clip's filmstrip, with the window of the strip its cut covers:
    /// where in the footage it starts and how much of the footage it spans,
    /// both as fractions, so the lane can pick the frame under each tile
    /// without knowing about speed or source time.
    fn strip_of(&self, clip: &Clip) -> StripData {
        if !matches!(clip.kind, model::ClipKind::Video | model::ClipKind::Image) {
            return StripData::default();
        }
        let Some(mut strip) = self.strips.get(&clip.media_id) else {
            return StripData::default();
        };
        let (mut start, mut span) = (0.0, 1.0);
        if let Some((cut_start, cut_span, duration)) = self.cut_of(clip) {
            start = cut_start;
            span = cut_span;
            // The cell strip, once it is here: the cut re-expressed as a
            // window of that cell rather than of the whole file. Until it
            // arrives the file's strip stands in, and the tiles repeat.
            if let Some((level, cell)) = strip_window(start, span, duration)
                && let Some(window) = self.windows.get(&window_key(&clip.media_id, level, cell))
            {
                let (cell_start, cell_span) = (window_start(level, cell), window_span(level));
                start = ((start - cell_start) / cell_span).clamp(0.0, 1.0);
                span = (span / cell_span).clamp(0.0, 1.0);
                strip = window;
            }
        }
        StripData {
            image: strip.image.clone(),
            frames: strip.frames,
            frame_width: strip.frame_width,
            height: strip.height,
            start: start as f32,
            span: span as f32,
        }
    }

    /// The memoised envelope for one clip, quantised to a thirtieth of a
    /// second so a trim revisits a handful of entries rather than minting
    /// one per pointer event, and to a step of columns so a zoom rebuilds
    /// it at each step of that and not at every pixel. A sound clip's
    /// whole body, a picture clip's band under its frames; a picture that
    /// is muted, or whose file has no sound, shows the empty band.
    ///
    /// At unity: the clip's volume scales the drawing in the lane, so a
    /// volume drag never comes here - see `format::wave_path`.
    ///
    /// Only the window of the clip that is on screen, and a screen either
    /// side of it, is built: what comes back is the path and where it
    /// sits on the clip, as fractions of the clip's length. A clip zoomed
    /// in far enough to be wider than the screen many times over would
    /// otherwise spread its bars across all of that width - a bar the
    /// width of a finger - or want tens of thousands of them. This way a
    /// bar is one pitch on screen at every zoom, and the count is bounded
    /// by the screen, not the clip. The window is quantised to strides
    /// of sixty-four bars, so a scroll rebuilds at each stride and not at
    /// every pixel, and the cache holds a few hundred windows before it
    /// is emptied.
    fn wave(&self, clip: &Clip) -> (SharedString, f32, f32) {
        if clip.muted == Some(true) {
            return Default::default();
        }
        // The stream this clip plays; its peaks come when they are decoded,
        // and until then the lane is bare rather than showing another
        // track's shape.
        let art = art_key(&clip.media_id, clip.audio_stream);
        let Some(peaks) = self.peaks.get(&art) else {
            return Default::default();
        };
        let step = |seconds: f32| (seconds * WAVE_STEPS).round() / WAVE_STEPS;
        let (source_start, duration) = (step(clip.source_start as f32), step(clip.duration as f32));
        if duration <= 0.0 {
            return Default::default();
        }
        let seconds_per_pixel = self.lanes.seconds_per_pixel;
        let stride = (WAVE_PITCH * seconds_per_pixel * 64.0).max(1.0 / WAVE_STEPS);
        let (from, to) = match self.lanes.published_span() {
            Some((left, right)) => {
                let from = (left - clip.start as f32).max(0.0);
                let to = (right - clip.start as f32).min(duration);
                if to > from {
                    (
                        (from / stride).floor() * stride,
                        ((to / stride).ceil() * stride).min(duration),
                    )
                } else {
                    (0.0, duration)
                }
            }
            None => (0.0, duration),
        };
        let window = to - from;
        let columns = wave_columns(window, seconds_per_pixel);
        let placed = (from / duration, window / duration);
        let key = format!("{art}|{source_start:.3}|{from:.3}|{window:.3}|{columns}");
        if let Some(cached) = self.waves.borrow().get(&key) {
            return (cached.clone(), placed.0, placed.1);
        }
        let built = SharedString::from(wave_path(
            peaks,
            source_start + from,
            window,
            columns,
            WAVE_BAR,
        ));
        let mut waves = self.waves.borrow_mut();
        if waves.len() >= 512 {
            waves.clear();
        }
        waves.insert(key, built.clone());
        (built, placed.0, placed.1)
    }

    // ── placing things ──

    /// What a library payload names - "media:12", "text:default:Title" -
    /// before the timeline has decided where to put it.
    pub fn incoming(&self, payload: &str) -> Option<DropPlan> {
        let mut fields = payload.splitn(3, ':');
        let (sort, id) = (fields.next()?, fields.next()?);
        let label = fields.next().unwrap_or(id);
        match sort {
            "media" => {
                let item = self.media.by_row(self.project(), id.parse().ok()?)?;
                Some(DropPlan {
                    kind: match item.kind {
                        model::MediaKind::Audio => ClipKind::Audio,
                        model::MediaKind::Image => ClipKind::Image,
                        model::MediaKind::Video => ClipKind::Video,
                    },
                    label: item.name.clone(),
                    media: item.id.clone(),
                    start: 0.0,
                    // A still has no length of its own; a file with no stated
                    // duration gets the engine's own fallback.
                    duration: if item.kind == model::MediaKind::Image {
                        LAYER_DURATION
                    } else {
                        item.duration.unwrap_or(5.0) as f32
                    }
                    .max(MIN_DURATION),
                    row: 0,
                })
            }
            // A title: the preset's id rides where a file's media id would,
            // "default" for the plain one.
            "text" => Some(DropPlan {
                kind: ClipKind::Text,
                label: label.to_owned(),
                media: id.to_owned(),
                start: 0.0,
                duration: LAYER_DURATION,
                row: 0,
            }),
            // A figure from the Stickers page: the shape's name rides
            // where a file's media id would, and the label is what the
            // lane will call it.
            "shape" => Some(DropPlan {
                kind: ClipKind::Shape,
                label: label.to_owned(),
                media: model::ShapeKind::parse(id)?.id().to_owned(),
                start: 0.0,
                duration: LAYER_DURATION,
                row: 0,
            }),
            // A look dragged from the Filters page: a layer over a span.
            // The package id rides in `media`, there being no file.
            "filter" => Some(DropPlan {
                kind: ClipKind::Filter,
                label: label.to_owned(),
                media: id.to_owned(),
                start: 0.0,
                duration: LAYER_DURATION,
                row: 0,
            }),
            _ => None,
        }
    }

    /// A drag over the lanes: the pointer names the moment and the lane.
    pub fn plan(&self, payload: &str, seconds: f32, row: i32) -> Option<DropPlan> {
        let mut plan = self.incoming(payload)?;
        let lanes = self.timeline().tracks.len() as i32;
        if lanes == 0 {
            return None;
        }
        plan.row = row.clamp(0, lanes - 1);
        if self
            .row_track(plan.row)
            .is_none_or(|track| self.locked(&track.id))
        {
            return None;
        }
        plan.start = self
            .snapped(seconds.max(0.0), 8.0 * self.lanes.seconds_per_pixel, "")
            .max(0.0);
        Some(plan)
    }

    /// Commits a plan that has a lane, and selects what it made.
    pub fn place(&mut self, plan: &DropPlan) {
        let Some(track_id) = self.row_track(plan.row).map(|track| track.id.clone()) else {
            return;
        };
        let created = if plan.kind == ClipKind::Text {
            self.add_title(
                Some(track_id),
                f64::from(plan.start),
                f64::from(plan.duration),
                &plan.media,
            )
        } else if plan.kind == ClipKind::Shape {
            self.add_shape(
                Some(track_id),
                f64::from(plan.start),
                f64::from(plan.duration),
                &plan.media,
                &plan.label,
            )
        } else if plan.kind == ClipKind::Filter {
            self.apply(Command::AddLayerClip {
                track_id: Some(track_id),
                start: f64::from(plan.start),
                duration: Some(f64::from(plan.duration)),
                effect_id: plan.media.clone(),
                name: plan.label.clone(),
            })
        } else {
            self.apply(Command::AddClip {
                media_id: plan.media.clone(),
                track_id,
                start: f64::from(plan.start),
                ripple: true,
            })
        };
        if let Some(id) = created {
            self.selection = vec![id];
        }
    }

    /// A card clicked rather than dragged: the playhead names the moment
    /// and the engine finds a lane with room.
    pub fn place_at_playhead(&mut self, payload: &str) {
        let Some(plan) = self.incoming(payload) else {
            return;
        };
        let start = f64::from(self.playhead.max(0.0));
        let created = if plan.kind == ClipKind::Text {
            self.add_title(None, start, f64::from(plan.duration), &plan.media)
        } else if plan.kind == ClipKind::Shape {
            self.add_shape(
                None,
                start,
                f64::from(plan.duration),
                &plan.media,
                &plan.label,
            )
        } else if plan.kind == ClipKind::Filter {
            self.apply(Command::AddLayerClip {
                track_id: None,
                start,
                duration: Some(f64::from(plan.duration)),
                effect_id: plan.media.clone(),
                name: plan.label.clone(),
            })
        } else {
            self.apply(Command::AddClipAtFirstFree {
                media_id: plan.media.clone(),
                start,
            })
        };
        if let Some(id) = created {
            self.selection = vec![id];
        }
    }

    /// Places a title in the look a preset names - "default" for the plain
    /// title - and, when the preset brings a font, registers the font on
    /// the project in the same step, so the painter finds it the first
    /// time it draws the words.
    fn add_title(
        &mut self,
        track_id: Option<String>,
        start: f64,
        duration: f64,
        preset: &str,
    ) -> Option<String> {
        let (style, offset_y, font) = match self.text_presets.iter().find(|held| held.id == preset)
        {
            Some(found) => (
                found.style.clone(),
                found.offset_y,
                presets::install_font(&self.host.dirs, found),
            ),
            None => (new_title_style(&self.prefs), None, None),
        };
        // Over the picture, not under it: the first free lane above every
        // occupied one, minted at the top when there is none. A drop onto a
        // lane names its track and is placed there regardless.
        let add = Command::AddTextClip {
            track_id,
            start,
            style: Some(style),
            duration: Some(duration),
            offset_y,
            above: true,
        };
        match font {
            Some((family, path)) => self.apply(Command::Batch {
                commands: vec![Command::AddFont { family, path }, add],
            }),
            None => self.apply(add),
        }
    }

    /// A shape from the Stickers page, placed the way a title is: over
    /// the picture, on the first free lane above it.
    fn add_shape(
        &mut self,
        track_id: Option<String>,
        start: f64,
        duration: f64,
        shape: &str,
        name: &str,
    ) -> Option<String> {
        let kind = model::ShapeKind::parse(shape).unwrap_or_default();
        self.apply(Command::AddShapeClip {
            track_id,
            above: true,
            start,
            style: Some(model::ShapeStyle::of(kind)),
            duration: Some(duration),
            name: name.to_owned(),
        })
    }

    /// The selected clip's id, when exactly one is selected.
    /// The clips the inspector's knobs write to: the one selected clip, or
    /// every selected clip when two or more are selected and all of them
    /// are titles - a batch of captions styled together. Anything else
    /// selected together is nothing the inspector edits.
    /// https://github.com/jub0t/Concat/issues/224
    pub fn edit_targets(&self) -> Vec<String> {
        if let Some(id) = self.sole_selection() {
            return vec![id];
        }
        if self.selection.len() < 2 {
            return Vec::new();
        }
        let titles = self.selection.iter().all(|id| {
            self.clip(id)
                .is_some_and(|clip| clip.kind == model::ClipKind::Text)
        });
        if titles {
            self.selection.clone()
        } else {
            Vec::new()
        }
    }

    pub fn sole_selection(&self) -> Option<String> {
        (self.selection.len() == 1).then(|| self.selection[0].clone())
    }

    /// The look being shown over the picture, by id, while one is.
    pub fn audition_of(&self) -> Option<&str> {
        self.audition.as_deref()
    }

    /// Shows a look over the whole picture without laying it down: what a
    /// single click on a Filters card does. The same card clicked again
    /// takes it off; a double-click or the card's plus lays the layer.
    pub fn audition_catalogue(&mut self, id: &str) {
        if self.session.is_none() {
            return;
        }
        let same = self.audition.as_deref() == Some(id);
        if !same && self.audition.is_none() {
            self.notify(&t("studio.showingLookOverPicture"), false);
        }
        self.audition = (!same).then(|| id.to_owned());
        self.request_preview();
    }

    /// Lays a filter down as a layer at the playhead - what the Filters
    /// page's double-click and plus do - and ends the showing, the layer
    /// now being on the timeline to see.
    pub fn place_filter_layer(&mut self, id: &str, label: &str) {
        self.audition = None;
        self.place_at_playhead(&format!("filter:{id}:{label}"));
    }

    /// A catalogue filter or effect, applied to the selected clip's chain.
    pub fn apply_catalogue(&mut self, id: &str, video: bool) {
        let Some(clip_id) = self.sole_selection() else {
            self.notify(&t("studio.selectClipTimelineFirst"), true);
            return;
        };
        let Some(clip) = self.clip(&clip_id).cloned() else {
            return;
        };
        if video
            && !(clip.kind.is_visual()
                || matches!(clip.kind, model::ClipKind::Text | model::ClipKind::Shape))
        {
            self.notify(&t("studio.selectVideoImageText"), true);
            return;
        }
        if !video && clip.kind == model::ClipKind::Image {
            self.notify("A still has no sound to filter", true);
            return;
        }
        let entry = AppliedFilter::new(id);
        let patch = if video {
            let mut effects = clip.video_effects.clone();
            effects.push(entry);
            ClipPatch {
                video_effects: Some(effects),
                ..ClipPatch::default()
            }
        } else {
            let mut filters = clip.filters.clone();
            filters.push(entry);
            ClipPatch {
                filters: Some(filters),
                ..ClipPatch::default()
            }
        };
        self.apply(Command::UpdateClip { clip_id, patch });
        // Show it: the applied chain is where the knobs are, and a card
        // that did something with no visible result reads as a card that
        // did nothing.
        self.inspector_jump = (
            self.inspector_jump.0 + 1,
            if video { "Effects" } else { "Audio" },
            if video { "Effects" } else { "Sound" },
        );
    }

    /// The clip that ends where `clip` starts, on the same track - the
    /// outgoing side of the cut a transition into `clip` rides. `None` when
    /// nothing is adjacent, which is when a transition has nothing to
    /// dissolve from.
    fn outgoing_of(&self, clip: &Clip) -> Option<&Clip> {
        let frame = self.frame_seconds();
        self.timeline()
            .clips
            .iter()
            .find(|other| {
                other.id != clip.id
                    && other.track_id == clip.track_id
                    && (other.start + other.duration - clip.start).abs() < frame / 2.0
            })
            .map(|other| other.as_ref())
    }

    /// One frame of the active timeline, in seconds - the same tolerance
    /// `concat-export`'s own adjacency test uses, so a clip this call finds
    /// adjacent is one the exporter will too.
    fn frame_seconds(&self) -> f64 {
        let video = &self.timeline().video;
        video.rate_den as f64 / video.rate_num.max(1) as f64
    }

    /// The duration a transition into `clip` can actually run at: the
    /// requested length, clamped to both clips' own durations. `None` when there
    /// is no adjacent clip to dissolve from, or both clips are shorter than a frame -
    /// the ways a transition here would render as nothing at all.
    pub fn transition_duration(&self, clip: &Clip, requested: f64) -> Option<f64> {
        let outgoing = self.outgoing_of(clip)?;
        let frame = self.frame_seconds();
        let max_duration = outgoing.duration.min(clip.duration);
        if max_duration < frame {
            return None;
        }
        let requested = if requested <= 0.0 {
            frame
        } else {
            requested.max(frame)
        };
        let duration = requested.min(max_duration).max(0.0);
        (duration >= frame).then_some(duration)
    }

    /// The cut a transition from the library goes on. Two clips selected
    /// that meet on one track name the cut between them - the simple way,
    /// and the one the Transitions page asks for. One clip selected names
    /// the cut into it, for a transition picked after a click on the lane.
    /// The incoming clip's id, or what to say instead.
    fn transition_cut(&self) -> Result<String, String> {
        match self.selection.as_slice() {
            [one] => Ok(one.clone()),
            [a, b] => {
                let (Some(a), Some(b)) = (self.clip(a), self.clip(b)) else {
                    return Err(t("studio.selectTwoAdjacentClips"));
                };
                let (first, second) = if a.start <= b.start { (a, b) } else { (b, a) };
                if self
                    .outgoing_of(second)
                    .is_some_and(|outgoing| outgoing.id == first.id)
                {
                    Ok(second.id.clone())
                } else {
                    Err(t("studio.clipsDoNotMeet"))
                }
            }
            _ => Err(t("studio.selectTwoAdjacentClips")),
        }
    }

    /// Puts the catalogue transition `id` on the cut the selection names;
    /// see [`Studio::transition_cut`].
    pub fn apply_transition(&mut self, id: &str) {
        let clip_id = match self.transition_cut() {
            Ok(clip_id) => clip_id,
            Err(why) => {
                self.notify(&why, true);
                return;
            }
        };
        let Some(clip) = self.clip(&clip_id) else {
            return;
        };
        // One clip with nothing meeting it has no cut to put a transition
        // on: the answer is the other clip, not a longer one.
        let Some(outgoing) = self.outgoing_of(clip) else {
            self.notify(&t("studio.selectTwoAdjacentClips"), true);
            return;
        };
        if !clip.kind.is_visual() || !outgoing.kind.is_visual() {
            self.notify(&t("studio.selectVideoImageClip"), true);
            return;
        }
        let Some(duration) = self.transition_duration(clip, 0.5) else {
            self.notify(&t("studio.noRoomForTransition"), true);
            return;
        };
        self.apply(Command::UpdateClip {
            clip_id,
            patch: ClipPatch {
                transition_in: Some(Some(Transition {
                    id: id.to_owned(),
                    duration,
                })),
                ..ClipPatch::default()
            },
        });
    }

    /// The clip whose incoming transition is selected: the one picked on
    /// its own while no clip is, if it still has a transition.
    pub fn selected_transition(&self) -> Option<&Clip> {
        if !self.selection.is_empty() {
            return None;
        }
        let id = self.transition_selected.as_deref()?;
        self.clip(id).filter(|clip| clip.transition_in.is_some())
    }

    /// The clip a transition verb acts on: the selected transition's, or
    /// the one selected clip's.
    fn transition_target(&self) -> Option<String> {
        self.selected_transition()
            .map(|clip| clip.id.clone())
            .or_else(|| self.sole_selection())
    }

    /// Takes the transition off its clip, leaving a plain cut.
    pub fn remove_transition(&mut self) {
        let Some(clip_id) = self.transition_target() else {
            return;
        };
        self.transition_selected = None;
        self.apply(Command::UpdateClip {
            clip_id,
            patch: ClipPatch {
                transition_in: Some(None),
                ..ClipPatch::default()
            },
        });
    }

    /// Sets the selected clip's transition to `seconds`, clamped to what
    /// the cut can actually support - the same ceiling `apply_transition`
    /// enforces, kept here so a drag on the timeline can never ask for more
    /// than the render will give it.
    pub fn set_transition_duration(&mut self, seconds: f64) {
        let Some(clip_id) = self.transition_target() else {
            return;
        };
        self.set_clip_transition_duration(&clip_id, seconds);
    }

    /// Sets `clip_id`'s transition to `seconds`, clamped to what the cut can
    /// actually support.
    pub fn set_clip_transition_duration(&mut self, clip_id: &str, seconds: f64) {
        let Some(clip) = self.clip(clip_id) else {
            return;
        };
        let Some(existing) = clip.transition_in.clone() else {
            return;
        };
        let Some(duration) = self.transition_duration(clip, seconds.max(0.0)) else {
            return;
        };
        self.apply(Command::UpdateClip {
            clip_id: clip_id.to_owned(),
            patch: ClipPatch {
                transition_in: Some(Some(Transition {
                    duration,
                    ..existing
                })),
                ..ClipPatch::default()
            },
        });
    }

    /// Drops one entry from the selected clip's chains, by the row id the
    /// inspector shows: video effects count from 1, audio filters from 1000.
    pub fn remove_effect(&mut self, row: i32) {
        let Some(clip_id) = self.sole_selection() else {
            return;
        };
        let Some(clip) = self.clip(&clip_id).cloned() else {
            return;
        };
        let patch = if row >= 1000 {
            let index = (row - 1000) as usize;
            let mut filters = clip.filters.clone();
            if index >= filters.len() {
                return;
            }
            filters.remove(index);
            ClipPatch {
                filters: Some(filters),
                ..ClipPatch::default()
            }
        } else {
            let index = (row - 1).max(0) as usize;
            let mut effects = clip.video_effects.clone();
            if index >= effects.len() {
                return;
            }
            effects.remove(index);
            ClipPatch {
                video_effects: Some(effects),
                ..ClipPatch::default()
            }
        };
        self.apply(Command::UpdateClip { clip_id, patch });
    }

    // ── the chains ──
    //
    // The picture's chain and the sound's, edited by index from the
    // inspector's stacks. Adding, removing, reordering and bypassing are
    // each one command; a knob is a stream and goes through the echo, the
    // way every other inspector control does.

    /// One edit to the selected clip's chain, as a command. `edit` returns
    /// false to say it changed nothing.
    fn chain_edit(&mut self, audio: bool, edit: impl FnOnce(&mut Vec<AppliedFilter>) -> bool) {
        let Some(clip_id) = self.sole_selection() else {
            return;
        };
        let Some(clip) = self.clip(&clip_id).cloned() else {
            return;
        };
        let mut chain = if audio {
            clip.filters
        } else {
            clip.video_effects
        };
        if !edit(&mut chain) {
            return;
        }
        let patch = if audio {
            ClipPatch {
                filters: Some(chain),
                ..ClipPatch::default()
            }
        } else {
            ClipPatch {
                video_effects: Some(chain),
                ..ClipPatch::default()
            }
        };
        self.apply(Command::UpdateClip { clip_id, patch });
    }

    pub fn chain_add(&mut self, audio: bool, id: &str) {
        self.apply_catalogue(id, !audio);
    }

    pub fn chain_toggle(&mut self, audio: bool, index: i32) {
        self.chain_edit(audio, |chain| {
            let Some(entry) = usize::try_from(index).ok().and_then(|i| chain.get_mut(i)) else {
                return false;
            };
            entry.enabled = !entry.enabled;
            true
        });
    }

    pub fn chain_move(&mut self, audio: bool, index: i32, delta: i32) {
        self.chain_edit(audio, |chain| {
            let Ok(from) = usize::try_from(index) else {
                return false;
            };
            let Ok(to) = usize::try_from(index + delta) else {
                return false;
            };
            if from >= chain.len() || to >= chain.len() || from == to {
                return false;
            }
            chain.swap(from, to);
            true
        });
    }

    pub fn chain_remove(&mut self, audio: bool, index: i32) {
        self.chain_edit(audio, |chain| {
            let Ok(at) = usize::try_from(index) else {
                return false;
            };
            if at >= chain.len() {
                return false;
            }
            chain.remove(at);
            true
        });
    }

    /// One knob of one link, on the echo. `clip_commit` makes it real.
    pub fn chain_set_param(&mut self, audio: bool, index: i32, key: &str, value: f32) {
        let Some(id) = self.sole_selection() else {
            return;
        };
        self.begin_echo();
        let Some(clip) = self.echo_clip_mut(&id) else {
            return;
        };
        let chain = if audio {
            &mut clip.filters
        } else {
            &mut clip.video_effects
        };
        if let Some(entry) = usize::try_from(index).ok().and_then(|i| chain.get_mut(i)) {
            entry.params.insert(key.to_owned(), f64::from(value));
        }
    }

    /// One knob of the colour panel, on the echo. The adjust package joins
    /// the head of the picture's chain the first time a knob moves, so an
    /// untouched clip carries nothing. A wheel's puck and master are knobs
    /// of their own here, `<key>.x`, `<key>.y` and `<key>.m`.
    pub fn adjust_set(&mut self, key: &str, value: f32) {
        let point = self.key_point().map(|(_, at)| at);
        self.edit_adjust_link(|link| match point {
            // A knob that rides is edited where the playhead is: the value
            // becomes the key there, put on if there was none. Setting the
            // constant under a ride would change nothing on screen.
            Some(at) if link.is_keyed(key) => {
                let ease = ease_before(link.keys_on(key), at);
                link.set_key(key, at, f64::from(value), ease);
            }
            _ => {
                link.params.insert(key.to_owned(), f64::from(value));
            }
        });
    }

    /// One point of one of the colour panel's curves, on the echo: moved,
    /// or put down when `point` is negative.
    pub fn adjust_curve_point(&mut self, key: &str, point: i32, x: f32, y: f32) {
        self.edit_adjust_link(|link| {
            grading::set_curve_point(&mut link.params, key, point, x.into(), y.into());
        });
    }

    /// One point of one of the colour panel's curves taken off, on the echo.
    pub fn adjust_curve_remove(&mut self, key: &str, point: i32) {
        self.edit_adjust_link(|link| grading::remove_curve_point(&mut link.params, key, point));
    }

    /// The selected clip's colour link on the echo, put at the head of the
    /// picture's chain if it is not there yet, handed to `edit`. Nothing
    /// for a clip that is not a picture.
    fn edit_adjust_link(&mut self, edit: impl FnOnce(&mut AppliedFilter)) {
        let Some(id) = self.sole_selection() else {
            return;
        };
        self.begin_echo();
        let Some(clip) = self.echo_clip_mut(&id) else {
            return;
        };
        if !clip.kind.is_visual() {
            return;
        }
        let entry = match clip
            .video_effects
            .iter()
            .position(|entry| entry.id == ADJUST_ID)
        {
            Some(entry) => entry,
            None => {
                clip.video_effects.insert(0, AppliedFilter::new(ADJUST_ID));
                0
            }
        };
        edit(&mut clip.video_effects[entry]);
    }

    /// One point of a curve of the picture chain's link `index`, on the
    /// echo: moved, or put down when `point` is negative.
    pub fn chain_curve_point(&mut self, index: i32, key: &str, point: i32, x: f32, y: f32) {
        self.edit_chain_link(index, |link| {
            grading::set_curve_point(&mut link.params, key, point, x.into(), y.into());
        });
    }

    /// One point of a curve of the picture chain's link `index` taken off,
    /// on the echo.
    pub fn chain_curve_remove(&mut self, index: i32, key: &str, point: i32) {
        self.edit_chain_link(index, |link| {
            grading::remove_curve_point(&mut link.params, key, point);
        });
    }

    /// A colour knob of the picture chain's link `index`, on the echo:
    /// packed into the knob's number (see `packed`).
    pub fn chain_set_colour(&mut self, index: i32, key: &str, colour: slint::Color) {
        self.edit_chain_link(index, |link| {
            link.params.insert(key.to_owned(), packed(colour));
        });
    }

    /// The picture chain's link `index` of the selected clip, on the echo.
    fn edit_chain_link(&mut self, index: i32, edit: impl FnOnce(&mut AppliedFilter)) {
        let Some(id) = self.sole_selection() else {
            return;
        };
        self.begin_echo();
        let Some(clip) = self.echo_clip_mut(&id) else {
            return;
        };
        if let Some(link) = usize::try_from(index)
            .ok()
            .and_then(|index| clip.video_effects.get_mut(index))
        {
            edit(link);
        }
    }

    /// The colour link of the selected clip - the one the Adjust panel's
    /// keys go on - by index, with the clip and the playhead's place in it.
    /// None when nothing is selected, it is not a picture, or the playhead
    /// is outside it.
    fn adjust_point(&self) -> Option<(Clip, f64, Option<usize>)> {
        let (clip, at) = self.key_point()?;
        if !clip.kind.is_visual() {
            return None;
        }
        let entry = clip
            .video_effects
            .iter()
            .position(|entry| entry.id == ADJUST_ID);
        Some((clip.clone(), at, entry))
    }

    /// Puts a key on one Adjust knob at the playhead, or takes off the one
    /// there. Like `toggle_key`, the new key holds what the knob is worth
    /// at that instant, so pressing the diamond never moves the picture. A
    /// wheel - named by itself or by one of its own keys - is one knob: its
    /// puck and master are keyed, and unkeyed, together.
    pub fn toggle_adjust_key(&mut self, key: &str) {
        let Some((clip, at, entry)) = self.adjust_point() else {
            return;
        };
        let names = adjust_key_names(key);
        let clip_id = clip.id.clone();
        let Some(entry) = entry else {
            // No colour link yet: one goes on, then the keys on it, as one
            // edit, so an undo takes all of it back.
            let mut effects = clip.video_effects.clone();
            effects.insert(0, AppliedFilter::new(ADJUST_ID));
            let mut commands = vec![Command::UpdateClip {
                clip_id: clip_id.clone(),
                patch: ClipPatch {
                    video_effects: Some(effects),
                    ..ClipPatch::default()
                },
            }];
            commands.extend(names.iter().map(|(name, default)| Command::SetEffectKey {
                clip_id: clip_id.clone(),
                entry: 0,
                key: name.clone(),
                at,
                value: *default,
                ease: model::KeyEase::LINEAR,
            }));
            self.apply(Command::Batch { commands });
            return;
        };
        let link = &clip.video_effects[entry];
        // The knob is keyed here when its own key - a wheel's master - is.
        let on = names
            .last()
            .is_some_and(|(name, _)| link.key_at(name, at).is_some());
        let mut commands: Vec<Command> = names
            .iter()
            .filter_map(|(name, default)| {
                if on {
                    link.key_at(name, at).map(|_| Command::ClearEffectKey {
                        clip_id: clip_id.clone(),
                        entry,
                        key: name.clone(),
                        at,
                    })
                } else {
                    Some(Command::SetEffectKey {
                        clip_id: clip_id.clone(),
                        entry,
                        key: name.clone(),
                        at,
                        value: link.value_at(name, at, *default),
                        ease: ease_before(link.keys_on(name), at),
                    })
                }
            })
            .collect();
        let command = if commands.len() == 1 {
            commands.remove(0)
        } else {
            Command::Batch { commands }
        };
        self.apply(command);
    }

    /// Takes every key off one Adjust knob - a wheel's puck and master
    /// together - leaving it the value it holds.
    pub fn clear_adjust_keys(&mut self, key: &str) {
        let Some((clip, _, Some(entry))) = self.adjust_point() else {
            return;
        };
        let link = &clip.video_effects[entry];
        let mut commands: Vec<Command> = adjust_key_names(key)
            .into_iter()
            .filter(|(name, _)| link.is_keyed(name))
            .map(|(name, _)| Command::ClearEffectKeys {
                clip_id: clip.id.clone(),
                entry,
                key: name,
            })
            .collect();
        match commands.len() {
            0 => {}
            1 => {
                self.apply(commands.remove(0));
            }
            _ => {
                self.apply(Command::Batch { commands });
            }
        }
    }

    /// Moves the playhead to an Adjust knob's previous (-1) or next (+1) key.
    pub fn step_adjust_key(&mut self, key: &str, delta: i32) {
        let Some((clip, at, Some(entry))) = self.adjust_point() else {
            return;
        };
        let Some((name, _)) = adjust_key_names(key).pop() else {
            return;
        };
        let (prev, next) = clip.video_effects[entry].keys_around(&name, at);
        let Some(target) = (if delta < 0 { prev } else { next }) else {
            return;
        };
        self.seek((clip.start + target * clip.duration) as f32);
    }

    // ── gestures ──

    /// A press resolves the selection before anything moves, so a drag that
    /// starts on an already-selected clip carries the whole set.
    pub fn clip_pressed(&mut self, id: &str, additive: bool, edge: i32) {
        let Some(clip) = self.clip(id).cloned() else {
            return;
        };
        if self.locked(&clip.track_id) {
            return;
        }
        // Whatever the inspector still holds for the clip selected until
        // now lands before the selection moves: a commit flushed after the
        // change would look for it on the newly selected clip and lose it.
        self.flush_commit();
        // A press on a transition's seam or handles selects the transition,
        // not the clip it rides into; a drag there still sets its length.
        if edge == 2
            && let Some(transition) = clip.transition_in.as_ref()
        {
            self.selection.clear();
            self.transition_selected = Some(id.to_owned());
            self.begin_echo();
            self.gesture = Gesture::TransitionResize {
                clip: id.to_owned(),
                original_duration: transition.duration as f32,
            };
            return;
        }
        self.transition_selected = None;
        let already = self.selection.iter().any(|held| held == id);
        self.selection = if additive {
            if already {
                self.selection
                    .iter()
                    .filter(|held| held.as_str() != id)
                    .cloned()
                    .collect()
            } else {
                let mut next = self.selection.clone();
                next.push(id.to_owned());
                next
            }
        } else if already {
            self.selection.clone()
        } else {
            vec![id.to_owned()]
        };

        self.begin_echo();
        if edge >= 0 && self.selection.len() <= 1 {
            self.gesture = Gesture::Trim {
                clip: id.to_owned(),
                edge: if edge == 0 { Edge::Start } else { Edge::End },
                start: clip.start as f32,
                duration: clip.duration as f32,
                source_start: clip.source_start as f32,
            };
            return;
        }
        // What moves is what is selected, and nothing else. A detached
        // sound is a clip of its own: it goes where it is dragged, and its
        // picture stays, as any two clips do. It used to travel with its
        // picture, which made Detach a split in name only (#265); the link
        // between them is kept for Reattach, not for the drag.
        let moving = if self.selection.iter().any(|held| held == id) {
            self.selection.clone()
        } else {
            vec![id.to_owned()]
        };
        let origins = moving
            .iter()
            .filter_map(|clip_id| {
                let clip = self.clip(clip_id)?;
                Some(MoveOrigin {
                    clip: clip.id.clone(),
                    start: clip.start as f32,
                    row: self.row_of(&clip.track_id),
                })
            })
            .collect();
        self.gesture = Gesture::Move {
            primary: id.to_owned(),
            origins,
            lanes: self.lane_heights(),
        };
    }

    /// The pointer moved: the echo follows it.
    pub fn clip_dragged(&mut self, seconds: f32, pixels: f32) {
        let gesture = std::mem::replace(&mut self.gesture, Gesture::None);
        match &gesture {
            Gesture::Move {
                primary,
                origins,
                lanes,
            } => {
                let Some(anchor) = origins.iter().find(|origin| &origin.clip == primary) else {
                    self.gesture = gesture;
                    return;
                };
                let threshold = 8.0 * self.lanes.seconds_per_pixel;
                let snapped = self.snapped(anchor.start + seconds, threshold, primary);
                let shift = snapped - anchor.start;
                let rows = nearest_row(lanes, row_top(lanes, anchor.row) + pixels) - anchor.row;
                let count = self.timeline().tracks.len() as i32;
                let moves: Vec<ClipMove> = origins
                    .iter()
                    .filter_map(|origin| {
                        let row = (origin.row + rows).clamp(0, count - 1);
                        let track_id = self
                            .row_track(row)
                            .filter(|track| !self.locked(&track.id))
                            .map(|track| track.id.clone())
                            .or_else(|| {
                                self.clip(&origin.clip).map(|clip| clip.track_id.clone())
                            })?;
                        Some(ClipMove {
                            clip_id: origin.clip.clone(),
                            start: f64::from((origin.start + shift).max(0.0)),
                            track_id,
                        })
                    })
                    .collect();
                // The drop's own rule, asked of the timeline as it was before
                // the drag: the echo shows where the clips will land, in the
                // nearest gap rather than on top of a clip, or back where
                // they started when they would cover each other.
                let landed = self.timeline().resolve_moves(&moves).unwrap_or_else(|| {
                    origins
                        .iter()
                        .filter_map(|origin| {
                            let clip = self.clip(&origin.clip)?;
                            Some(ClipMove {
                                clip_id: clip.id.clone(),
                                start: clip.start,
                                track_id: clip.track_id.clone(),
                            })
                        })
                        .collect()
                });
                for wanted in landed {
                    if let Some(clip) = self.echo_clip_mut(&wanted.clip_id) {
                        clip.start = wanted.start;
                        clip.track_id = wanted.track_id;
                    }
                }
            }
            Gesture::Trim {
                clip,
                edge,
                start,
                duration,
                source_start,
            } => {
                let (id, edge) = (clip.clone(), *edge);
                let (start, duration, source_start) = (*start, *duration, *source_start);
                let threshold = 8.0 * self.lanes.seconds_per_pixel;
                let speed = self.clip(&id).map_or(1.0, |clip| clip.speed as f32);
                // A plain trim stops at the neighbours, as the command will;
                // a magnetic one moves them, so the lane is all its room.
                let (floor, ceiling) = if self.prefs.magnetic {
                    (0.0, f32::INFINITY)
                } else {
                    self.timeline()
                        .room_around(&id)
                        .map_or((0.0, f32::INFINITY), |(before, after)| {
                            (before as f32, after as f32)
                        })
                };
                if edge == Edge::Start {
                    // The head cannot pass the tail, and cannot pull material
                    // out of a file that has none before the in-point.
                    let wanted = self.snapped(start + seconds, threshold, &id);
                    let limit = start + duration - MIN_DURATION;
                    let at = wanted.clamp(
                        (start - source_start / speed.max(0.01))
                            .max(floor)
                            .min(limit),
                        limit,
                    );
                    let delta = at - start;
                    if let Some(clip) = self.echo_clip_mut(&id) {
                        clip.start = f64::from(at);
                        clip.duration = f64::from(duration - delta);
                        clip.source_start = f64::from(source_start + delta * speed);
                    }
                    // Trim follow: the playhead on the new first frame.
                    if self.prefs.trim_follow {
                        self.seek(at);
                    }
                } else {
                    let wanted = self.snapped(start + duration + seconds, threshold, &id);
                    let at = wanted.min(ceiling).max(start + MIN_DURATION);
                    if let Some(clip) = self.echo_clip_mut(&id) {
                        clip.duration = f64::from(at - start);
                    }
                    // Trim follow: the playhead on the new last frame, one
                    // frame inside the edge - at the edge itself the clip
                    // has already ended and the monitor would show what
                    // comes after it.
                    if self.prefs.trim_follow {
                        let frame = 1.0 / self.frame_rate().max(1.0);
                        self.seek((at - frame).max(start));
                    }
                }
            }
            Gesture::TransitionResize {
                clip,
                original_duration,
            } => {
                let id = clip.clone();
                let original_duration = *original_duration;
                if let Some(clip_ref) = self.clip(&id).cloned() {
                    let wanted = (original_duration + seconds).max(0.0) as f64;
                    if let Some(duration) = self.transition_duration(&clip_ref, wanted) {
                        if let Some(echo_clip) = self.echo_clip_mut(&id) {
                            if let Some(t) = echo_clip.transition_in.as_mut() {
                                t.duration = duration;
                            }
                        }
                    }
                }
            }
            // A stage gesture is the monitor's; the lanes have nothing to
            // add to it.
            Gesture::None
            | Gesture::StageMove { .. }
            | Gesture::StageScale { .. }
            | Gesture::StageRotate { .. }
            | Gesture::StageStretch { .. }
            | Gesture::TextWidth { .. }
            | Gesture::TextHeight { .. }
            | Gesture::Draw { .. }
            | Gesture::Paint { .. } => {}
        }
        self.gesture = gesture;
    }

    /// The pointer let go: the whole gesture becomes one command.
    pub fn clip_released(&mut self) {
        let gesture = std::mem::replace(&mut self.gesture, Gesture::None);
        let Some(echo) = self.echo.as_ref() else {
            return;
        };
        match gesture {
            Gesture::Move { origins, .. } => {
                let moves: Vec<ClipMove> = origins
                    .iter()
                    .filter_map(|origin| {
                        let after = echo.active().clip(&origin.clip)?;
                        Some(ClipMove {
                            clip_id: origin.clip.clone(),
                            start: after.start,
                            track_id: after.track_id.clone(),
                        })
                    })
                    .collect();
                self.echo = None;
                if !moves.is_empty() {
                    self.apply(Command::MoveClips { moves });
                }
            }
            Gesture::Trim {
                clip,
                edge,
                start,
                duration,
                ..
            } => {
                let after = echo.active().clip(&clip).cloned();
                self.echo = None;
                let Some(after) = after else { return };
                let delta = match edge {
                    Edge::Start => after.start - f64::from(start),
                    Edge::End => after.duration - f64::from(duration),
                };
                if delta.abs() > 1e-6 {
                    let id = clip.clone();
                    self.apply(Command::TrimClip {
                        clip_id: clip,
                        edge: match edge {
                            Edge::Start => TrimEdge::Start,
                            Edge::End => TrimEdge::End,
                        },
                        delta,
                        // Magnetic: the lane closes behind the edge.
                        // https://github.com/jub0t/Concat/issues/106
                        ripple: self.prefs.magnetic,
                    });
                    // Trim follow: a magnetic head trim slides the clip back
                    // into the gap, and the playhead goes with its first
                    // frame.
                    if self.prefs.trim_follow && edge == Edge::Start {
                        if let Some(start) = self.clip(&id).map(|clip| clip.start) {
                            self.seek(start as f32);
                        }
                    }
                }
            }
            Gesture::TransitionResize {
                clip,
                original_duration,
            } => {
                let after = echo.active().clip(&clip).cloned();
                self.echo = None;
                let Some(after) = after else { return };
                let new_duration = after
                    .transition_in
                    .as_ref()
                    .map(|t| t.duration)
                    .unwrap_or(0.0);
                if (new_duration - f64::from(original_duration)).abs() > 1e-4 {
                    self.set_clip_transition_duration(&clip, new_duration);
                }
            }
            Gesture::None => {
                self.echo = None;
            }
            // Not a lane gesture: hand it back untouched, echo and all.
            other @ (Gesture::StageMove { .. }
            | Gesture::StageScale { .. }
            | Gesture::StageRotate { .. }
            | Gesture::StageStretch { .. }
            | Gesture::TextWidth { .. }
            | Gesture::TextHeight { .. }
            | Gesture::Draw { .. }
            | Gesture::Paint { .. }) => {
                self.gesture = other;
            }
        }
    }

    // ── the inspector ──

    /// One field of the selected clip, on the echo. `clip_commit` turns the
    /// accumulated edits into commands.
    pub fn clip_set(&mut self, field: ClipField, value: f32) {
        let ids = self.edit_targets();
        if ids.is_empty() {
            return;
        }
        self.commit_targets = ids.clone();
        self.begin_echo();
        let value = f64::from(value);
        // A batch keeps its shape: the knob shows the first clip's value,
        // and what it is turned to is read against that; see `batch_value`.
        let reference = (ids.len() > 1)
            .then(|| relative_property(field))
            .flatten()
            .and_then(|property| {
                self.clip(&ids[0])
                    .map(|clip| shown(clip, property, place_in(clip, self.playhead)))
            });
        for id in ids {
            // The media's tracks, read before the echo is borrowed: a row of
            // the Audio panel's list is a stream index of the file.
            let audio_tracks: Vec<u32> = if field == ClipField::AudioTrack {
                self.clip(&id)
                    .and_then(|clip| self.project().media_by_id(&clip.media_id))
                    .map(|item| item.audio_tracks.iter().map(|track| track.index).collect())
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            // Where the playhead sits in the clip: a keyed property is written
            // there, as a key.
            let at = self
                .clip(&id)
                .map(|clip| place_in(clip, self.playhead))
                .unwrap_or(0.0);
            let value = match (reference, relative_property(field)) {
                (Some(reference), Some(property)) => self
                    .clip(&id)
                    .map(|clip| batch_value(field, value, reference, shown(clip, property, at)))
                    .unwrap_or(value),
                _ => value,
            };
            let Some(clip) = self.echo_clip_mut(&id) else {
                continue;
            };
            let text = clip.text.get_or_insert_with(TextStyle::default);
            match field {
                ClipField::Scale => {
                    write_keyable(clip, model::KeyProperty::Scale, value.clamp(0.05, 8.0), at)
                }
                ClipField::AudioTrack => {
                    // The first row is the file's default and is stored as such,
                    // so a clip on the first track saves as every clip did before
                    // there were tracks to choose.
                    let row = value.max(0.0) as usize;
                    clip.audio_stream = (row > 0).then(|| audio_tracks.get(row).copied()).flatten();
                }
                ClipField::StretchX => clip.stretch_x = value.clamp(0.1, 10.0),
                ClipField::StretchY => clip.stretch_y = value.clamp(0.1, 10.0),
                ClipField::OffsetX => write_keyable(
                    clip,
                    model::KeyProperty::OffsetX,
                    value.clamp(-1.0, 1.0),
                    at,
                ),
                ClipField::OffsetY => write_keyable(
                    clip,
                    model::KeyProperty::OffsetY,
                    value.clamp(-1.0, 1.0),
                    at,
                ),
                ClipField::Rotation => write_keyable(
                    clip,
                    model::KeyProperty::Rotation,
                    value.clamp(-180.0, 180.0),
                    at,
                ),
                ClipField::Opacity => {
                    write_keyable(clip, model::KeyProperty::Opacity, value.clamp(0.0, 1.0), at)
                }
                ClipField::CutoutFeather => {
                    clip.cutout.get_or_insert_with(model::Cutout::auto).feather =
                        value.clamp(0.0, model::MAX_FEATHER);
                }
                ClipField::Volume => {
                    write_keyable(clip, model::KeyProperty::Volume, value.max(0.0), at)
                }
                ClipField::Speed => {
                    let speed = value.clamp(0.0625, 16.0);
                    clip.duration =
                        (clip.duration * clip.speed / speed).max(f64::from(MIN_DURATION));
                    clip.speed = speed;
                }
                ClipField::PreservePitch => clip.preserve_pitch = value != 0.0,
                ClipField::FlipH => clip.flip_h = value != 0.0,
                ClipField::FlipV => clip.flip_v = value != 0.0,
                ClipField::Blend => {
                    let mode = concat_core::Blend::ALL
                        .get(value.max(0.0) as usize)
                        .copied()
                        .unwrap_or_default();
                    clip.blend = if mode == concat_core::Blend::Normal {
                        String::new()
                    } else {
                        mode.name().to_owned()
                    };
                }
                ClipField::CropLeft
                | ClipField::CropTop
                | ClipField::CropRight
                | ClipField::CropBottom => {
                    let mut crop = clip.crop.unwrap_or_default();
                    let edge = match field {
                        ClipField::CropLeft => &mut crop.left,
                        ClipField::CropTop => &mut crop.top,
                        ClipField::CropRight => &mut crop.right,
                        _ => &mut crop.bottom,
                    };
                    *edge = value.clamp(0.0, 0.9);
                    let crop = crop.tidy();
                    clip.crop = (!crop.is_none()).then_some(crop);
                }
                ClipField::FadeIn => clip.fade_in = value.clamp(0.0, clip.duration / 2.0),
                ClipField::FadeOut => clip.fade_out = value.clamp(0.0, clip.duration / 2.0),
                ClipField::FontSize => text.font_size = value.clamp(0.01, 0.5),
                ClipField::FontWeight => text.font_weight = value.clamp(100.0, 900.0),
                ClipField::Italic => text.italic = value != 0.0,
                ClipField::TextOpacity => text.opacity = value.clamp(0.0, 1.0),
                ClipField::Align => {
                    text.align = match value as i32 {
                        0 => TextAlign::Left,
                        2 => TextAlign::Right,
                        _ => TextAlign::Center,
                    }
                }
                ClipField::StrokeWidth => text.stroke_width = value.clamp(0.0, 0.15),
                ClipField::Shadow => text.shadow = value != 0.0,
                ClipField::LineHeight => text.line_height = value.clamp(0.7, 2.5),
                ClipField::Tracking => text.tracking = value.clamp(-0.05, 0.3),
                ClipField::TextWidth => text.max_width = value.clamp(0.0, 2.0),
                ClipField::TextHeight => text.max_height = value.clamp(0.0, 2.0),
                // The stroke's opacity is its colour's alpha; see `hex_rgba`.
                ClipField::StrokeOpacity => {
                    let edge = colour_of(&text.stroke_color);
                    text.stroke_color = hex_rgba(slint::Color::from_argb_u8(
                        (value.clamp(0.0, 1.0) * 255.0).round() as u8,
                        edge.red(),
                        edge.green(),
                        edge.blue(),
                    ));
                }
                // The background's opacity is its colour's alpha the same way.
                // Never zero from the dial: fully transparent is how "no
                // background" is stored, and the switch is what takes it away.
                ClipField::BackgroundOpacity => {
                    let plate = colour_of(&text.background);
                    text.background = hex_with_alpha(slint::Color::from_argb_u8(
                        (value.clamp(0.01, 1.0) * 255.0).round() as u8,
                        plate.red(),
                        plate.green(),
                        plate.blue(),
                    ));
                }
                ClipField::BackgroundRadius => text.background_radius = value.clamp(0.0, 0.2),
                ClipField::BackgroundPaddingX => {
                    text.background_padding_x = value.clamp(0.0, 0.5);
                }
                ClipField::BackgroundPaddingY => {
                    text.background_padding_y = value.clamp(0.0, 0.5);
                }
            }
            // A media clip has no text; the placeholder must not linger.
            if clip.kind != model::ClipKind::Text {
                clip.text = None;
            }
        }
    }

    pub fn clip_set_text(&mut self, field: ClipTextField, value: &str) {
        let ids = self.edit_targets();
        if ids.is_empty() {
            return;
        }
        // The words are each title's own: a batch is styled together, never
        // made to say the same thing.
        if field == ClipTextField::Content && ids.len() > 1 {
            return;
        }
        // One of the app's own fonts goes on the project before the title
        // is set in it, as a command of its own: the painter reads the
        // project's fonts, and the echo below is not the project.
        if field == ClipTextField::FontFamily {
            self.ensure_font(value);
        }
        self.commit_targets = ids.clone();
        self.begin_echo();
        for id in ids {
            let Some(clip) = self.echo_clip_mut(&id) else {
                continue;
            };
            if clip.kind != model::ClipKind::Text {
                continue;
            }
            let text = clip.text.get_or_insert_with(TextStyle::default);
            match field {
                ClipTextField::Content => {
                    text.content = value.to_owned();
                    let first = value.lines().next().unwrap_or("").trim().to_owned();
                    clip.name = if first.is_empty() {
                        "Title".into()
                    } else {
                        first
                    };
                }
                ClipTextField::FontFamily => text.font_family = value.to_owned(),
                _ => {}
            }
        }
        // The words are on the echo now; show them. A title being typed is
        // painted in memory at the monitor's size, the way a grip drag is,
        // so the picture keeps up with the keystrokes while the commit
        // still lands once, on the way out of the field.
        //
        // Pending from the first keystroke, not from the field's blur: the
        // echo is dropped by anything that changes the edit - a press on a
        // clip, an undo, a command from the menu - and `flush_commit` only
        // lands what is pending. Words typed and not yet blurred were on
        // the echo alone, and went with it. Nothing starts the timer here;
        // the blur does, and a flush before then lands them too.
        self.commit_pending = true;
        self.request_preview();
    }

    // ── fonts ──

    /// The families a title can be set in, in the order the picker shows
    /// them: the bundled face; then the user's own - the project's fonts
    /// and the app's installed ones, once each, in alphabetical order; then
    /// the system's, likewise, less any already listed.
    pub fn font_families(&self) -> Vec<String> {
        let mut out = vec![concat_text::BUNDLED_FAMILY.to_owned()];
        let mut own: Vec<String> = self
            .installed_fonts
            .iter()
            .map(|(family, _)| family.clone())
            .chain(
                self.session
                    .as_ref()
                    .map(|session| session.project().fonts.clone())
                    .unwrap_or_default()
                    .into_iter()
                    .map(|font| font.family),
            )
            .filter(|family| !family.is_empty())
            .collect();
        own.sort_by_cached_key(|family| family.to_lowercase());
        own.dedup();
        let mut held: std::collections::HashSet<String> = out
            .iter()
            .map(|family| family.to_ascii_lowercase())
            .collect();
        for family in own.into_iter().chain(self.system_fonts.iter().cloned()) {
            if held.insert(family.to_ascii_lowercase()) {
                out.push(family);
            }
        }
        out
    }

    /// What [`Studio::font_families`] is made of, hashed: the same number
    /// for the same list, without building it.
    fn fonts_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.installed_fonts.hash(&mut hasher);
        if let Some(session) = &self.session {
            for font in &session.project().fonts {
                font.family.hash(&mut hasher);
            }
        }
        self.system_fonts.hash(&mut hasher);
        hasher.finish()
    }

    /// The machine's families have been read; see `system_fonts`.
    pub fn system_fonts_known(&mut self, families: Vec<String>) {
        self.system_fonts = families;
    }

    /// Font files the user picked: each is copied to the app's own folder,
    /// so it is there next session and for every project, and registered
    /// on this project; the selected title is set in the first of them.
    pub fn import_fonts(&mut self, paths: Vec<std::path::PathBuf>) {
        log::info!("fonts: importing {} file(s)", paths.len());
        let mut first: Option<String> = None;
        for path in paths {
            let families = concat_text::families_in(&path);
            if families.is_empty() {
                self.notify(&tf("studio.notAFont", &[&path.display()]), true);
                continue;
            }
            let Some(installed) = presets::install_font_file(&self.host.dirs, &path) else {
                self.notify(&tf("studio.notAFont", &[&path.display()]), true);
                continue;
            };
            let file = installed.to_string_lossy().into_owned();
            for family in families {
                if !self.installed_fonts.iter().any(|(held, _)| *held == family) {
                    self.installed_fonts.push((family.clone(), file.clone()));
                }
                self.ensure_font(&family);
                first.get_or_insert(family);
            }
        }
        if let Some(family) = first {
            self.notify(&tf("studio.fontImported", &[&family]), false);
            if self.sole_selection().is_some() {
                self.clip_set_text(ClipTextField::FontFamily, &family);
                self.clip_commit();
            }
        }
    }

    /// Registers the file behind one of the app's installed families on the
    /// project, once, so the painter - which reads a project's fonts and
    /// the system's, and nothing else - finds it. Nothing for a family the
    /// project already carries, or one that is not the app's to register.
    fn ensure_font(&mut self, family: &str) {
        let family = family.trim().trim_matches('"');
        let carried = self.session.as_ref().is_some_and(|session| {
            session
                .project()
                .fonts
                .iter()
                .any(|font| font.family == family)
        });
        if carried {
            return;
        }
        let Some((_, path)) = self.installed_fonts.iter().find(|(held, _)| held == family) else {
            return;
        };
        let (family, path) = (family.to_owned(), path.clone());
        self.apply(Command::AddFont { family, path });
    }

    pub fn clip_set_colour(&mut self, field: ClipTextField, value: slint::Color) {
        let ids = self.edit_targets();
        if ids.is_empty() {
            return;
        }
        self.commit_targets = ids.clone();
        self.begin_echo();
        for id in ids {
            let Some(clip) = self.echo_clip_mut(&id) else {
                continue;
            };
            if clip.kind != model::ClipKind::Text {
                continue;
            }
            let text = clip.text.get_or_insert_with(TextStyle::default);
            match field {
                ClipTextField::Color => text.color = hex_of(value),
                ClipTextField::StrokeColor => text.stroke_color = hex_rgba(value),
                ClipTextField::Background => text.background = hex_with_alpha(value),
                _ => {}
            }
        }
    }

    /// The selected title's look becomes the one new titles start from,
    /// and the one generated captions are set in: its family, weight,
    /// colours, stroke, shadow and background, never its words. Kept in
    /// the preferences, so it outlives the project.
    /// https://github.com/jub0t/Concat/issues/224
    pub fn use_title_style_for_new(&mut self) {
        let Some(mut style) = self
            .edit_targets()
            .first()
            .and_then(|id| self.clip(id))
            .filter(|clip| clip.kind == model::ClipKind::Text)
            .and_then(|clip| clip.text.clone())
        else {
            return;
        };
        style.content.clear();
        self.prefs.title_style = Some(style);
        self.prefs.save(&self.host.dirs);
        self.notify(&t("textPanel.newTitlesUseThisStyle"), false);
    }

    /// The inspector's gesture is over: what differs between the echo and
    /// the session becomes commands, as one batch.
    /// An inspector control changed something on the echo and wants it
    /// committed. Held, not applied: a knob commits on every move, and the
    /// command, the mix and the publish happen once the moves pause. The
    /// monitor follows the echo in the meantime.
    pub fn clip_commit(&mut self) {
        self.commit_pending = true;
        self.request_preview();
        self.commit_timer.start(
            slint::TimerMode::SingleShot,
            std::time::Duration::from_millis(120),
            || {
                crate::host::Shell::with(|shell, app| {
                    shell.studio.borrow_mut().flush_commit();
                    shell.studio.borrow().publish(&app, &shell.models);
                });
            },
        );
    }

    /// Lands the held commit now, if there is one. Called before anything
    /// that would drop the echo it lives on - a command from elsewhere, an
    /// undo, a gesture on the lanes or the stage.
    pub fn flush_commit(&mut self) {
        if !self.commit_pending {
            return;
        }
        self.commit_pending = false;
        self.commit_timer.stop();
        self.commit_now();
    }

    fn commit_now(&mut self) {
        // The clips the echo was written for, whatever is selected now; see
        // `commit_targets`. The selection is the fallback for a commit asked
        // for with nothing written, which has nothing to land anyway.
        let mut ids = std::mem::take(&mut self.commit_targets);
        if ids.is_empty() {
            ids = self.sole_selection().into_iter().collect();
        }
        let Some(echo) = self.echo.take() else {
            return;
        };
        let mut commands = Vec::new();
        for id in &ids {
            let (Some(after), Some(before)) = (
                echo.active().clip(id),
                self.session
                    .as_ref()
                    .and_then(|session| session.project().active().clip(id)),
            ) else {
                continue;
            };
            commands.extend(clip_commands(id, before, after));
        }
        if commands.is_empty() {
            return;
        }
        // One undo step per gesture, not per pointer move: a commit that
        // changes the same things on the same clips as the last, within a
        // moment of it, folds into the last one's step. The editor does the
        // folding; this decides when a pause is long enough to be a new
        // gesture on the same knob.
        let key = format!("{}:{}", ids.join("+"), commit_key(&commands));
        let now = std::time::Instant::now();
        let continues = self
            .last_commit
            .as_ref()
            .is_some_and(|(last, at)| *last == key && now.duration_since(*at).as_millis() < 900);
        if !continues && let Some(session) = self.session.as_mut() {
            session.end_gesture();
        }
        let command = match commands.len() {
            1 => commands.remove(0),
            _ => Command::Batch { commands },
        };
        self.apply_within(&key, command);
        self.last_commit = Some((key, now));
    }

    // ── the stage ──
    //
    // The monitor as a place to edit, not only to look: the pictures under
    // the playhead can be grabbed, slid, scaled and turned where they are
    // drawn. The same echo-then-command shape as the lanes and the
    // inspector, with one difference: the echo is composited too, so the
    // frame follows the pointer and not only the outline does.

    /// Where `clip` lands in the output frame. The compositor's own rule —
    /// contain-fitted, then the transform — restated in fractions.
    pub fn footprint(&self, clip: &Clip) -> Footprint {
        let (width, height) = self.output_size();
        let (width, height) = (f64::from(width.max(1)), f64::from(height.max(1)));
        let painted = matches!(clip.kind, model::ClipKind::Text | model::ClipKind::Shape)
            .then(|| self.title_blocks.get(&clip.id).copied())
            .flatten();
        let (w, h) = if let Some(((bw, bh), _)) = painted {
            // The painter said how big the block came out.
            (
                f64::from(bw) * clip.scale / width,
                f64::from(bh) * clip.scale / height,
            )
        } else if clip.kind == model::ClipKind::Text {
            // Not painted yet: a reading of the style until it is. An em is
            // `font_size` of the frame's height, a glyph runs about six
            // tenths of one, and a little air is left around the block the
            // way the plate would.
            let text = clip.text.clone().unwrap_or_default();
            let lines: Vec<&str> = text.content.lines().collect();
            let rows = lines.len().max(1) as f64;
            let longest = lines
                .iter()
                .map(|line| line.chars().count())
                .max()
                .unwrap_or(0)
                .max(1) as f64;
            let em = text.font_size.max(0.005) * height;
            let glyph = 0.6 * em + text.tracking * height;
            (
                (longest * glyph + 0.6 * em) * clip.scale / width,
                (rows * em * text.line_height.max(0.5) + 0.5 * em) * clip.scale / height,
            )
        } else if clip.kind == model::ClipKind::Shape {
            // Not painted yet: the figure's box from its style, the way the
            // painter sizes it, until the painter says.
            let shape = clip.shape.clone().unwrap_or_default();
            let side = shape.size.clamp(0.01, 2.0) * height;
            let (w, h) = match shape.kind {
                model::ShapeKind::Square | model::ShapeKind::Circle => (side, side),
                model::ShapeKind::Triangle => (side, side * 0.866),
                model::ShapeKind::Parallelogram | model::ShapeKind::Trapezoid => (side, side * 0.6),
                model::ShapeKind::Line => (side, shape.stroke_width * height),
                model::ShapeKind::Arrow => (side, (shape.stroke_width * 4.0).max(0.18) * side),
            };
            (w * clip.scale / width, h * clip.scale / height)
        } else {
            let media = self.project().media_by_id(&clip.media_id);
            let source = media
                .and_then(|item| Some((item.width?, item.height?)))
                .filter(|(w, h)| *w > 0 && *h > 0)
                .map(|(w, h)| (f64::from(w), f64::from(h)))
                // What is left after the crop is what gets fitted.
                .map(|(w, h)| match clip.crop {
                    Some(crop) => (
                        w * (1.0 - crop.left - crop.right).max(0.1),
                        h * (1.0 - crop.top - crop.bottom).max(0.1),
                    ),
                    None => (w, h),
                });
            match source {
                Some((sw, sh)) => {
                    let fit = (width / sw).min(height / sh);
                    (
                        sw * fit * clip.scale / width,
                        sh * fit * clip.scale / height,
                    )
                }
                // Dimensions never learnt: the frame's own shape, which is
                // what the compositor falls back to as well.
                None => (clip.scale, clip.scale),
            }
        };
        // Then pulled along each axis, as the compositor pulls it - a
        // picture, that is. A title's box is its style's, and the compositor
        // never stretches one; see titles.rs.
        // https://github.com/jub0t/Concat/issues/119
        let (w, h) = if clip.kind == model::ClipKind::Text {
            (w, h)
        } else {
            (w * clip.stretch_x, h * clip.stretch_y)
        };
        // The placement at the playhead: the clip's own, or its ride's
        // where a property is keyed, so the box follows the keys.
        let at = place_in(clip, self.playhead);
        let base = concat_core::timeline::Transform {
            scale: shown(clip, model::KeyProperty::Scale, at),
            offset_x: shown(clip, model::KeyProperty::OffsetX, at),
            offset_y: shown(clip, model::KeyProperty::OffsetY, at),
            rotation: shown(clip, model::KeyProperty::Rotation, at),
            stretch_x: clip.stretch_x,
            stretch_y: clip.stretch_y,
        };
        let placed = base;
        let factor = placed.scale / clip.scale.max(1e-6);
        // A title anchored on an edge paints its block beside the clip's
        // position, not on it; the box goes where the block is. The offset
        // is canvas pixels, so it scales, stretches and turns with the clip
        // before it becomes a fraction of the frame.
        let (ax, ay) = match painted {
            Some((_, (dx, dy))) if dx != 0 || dy != 0 => {
                let px = f64::from(dx) * placed.scale * clip.stretch_x;
                let py = f64::from(dy) * placed.scale * clip.stretch_y;
                let (sin, cos) = placed.rotation.to_radians().sin_cos();
                (
                    (px * cos - py * sin) / width,
                    (px * sin + py * cos) / height,
                )
            }
            _ => (0.0, 0.0),
        };
        Footprint {
            cx: 0.5 + placed.offset_x + ax,
            cy: 0.5 + placed.offset_y + ay,
            w: w * factor,
            h: h * factor,
            rotation: placed.rotation,
        }
    }

    /// The pictures under the playhead on lanes that are showing, bottom of
    /// the stack first — the order the compositor lays them down, and the
    /// order the overlay draws them in.
    fn stage_clips(&self) -> Vec<&Clip> {
        let timeline = self.timeline();
        let playhead = f64::from(self.playhead);
        let showing: HashSet<&str> = timeline
            .tracks
            .iter()
            .filter(|track| track.visible)
            .map(|track| track.id.as_str())
            .collect();
        let mut clips: Vec<&Clip> = timeline
            .clips
            .iter()
            .map(|clip| clip.as_ref())
            .filter(|clip| {
                (clip.kind.is_visual()
                    || matches!(clip.kind, model::ClipKind::Text | model::ClipKind::Shape))
                    && showing.contains(clip.track_id.as_str())
                    && clip.start <= playhead
                    && playhead < clip.start + clip.duration
            })
            .collect();
        // Rows count from the top; the bottom of the stack is the highest.
        clips.sort_by_key(|clip| std::cmp::Reverse(self.row_of(&clip.track_id)));
        clips
    }

    pub fn stage_items(&self) -> Vec<StageItemData> {
        self.stage_clips()
            .into_iter()
            .map(|clip| {
                let footprint = self.footprint(clip);
                StageItemData {
                    id: clip.id.as_str().into(),
                    kind: kind_of(clip),
                    selected: self.selection.iter().any(|id| id == &clip.id),
                    cx: footprint.cx as f32,
                    cy: footprint.cy as f32,
                    w: footprint.w as f32,
                    h: footprint.h as f32,
                    rotation: footprint.rotation as f32,
                    scale: clip.scale as f32,
                }
            })
            .collect()
    }

    /// The topmost picture under a frame point, skipping locked lanes the
    /// way a press on the lanes does.
    fn stage_hit(&self, x: f64, y: f64) -> Option<String> {
        let frame = self.output_size();
        self.stage_clips()
            .into_iter()
            .rev()
            .filter(|clip| !self.locked(&clip.track_id))
            .find(|clip| self.footprint(clip).contains(x, y, frame))
            .map(|clip| clip.id.clone())
    }

    /// A press on the stage floor: resolve the selection, then arm a move
    /// of everything selected that is under the playhead. Empty stage
    /// clears the selection, as an empty lane does.
    pub fn stage_pressed(&mut self, x: f32, y: f32, additive: bool) {
        if self.calibrate_press(x, y) {
            return;
        }
        if self.mark_press(x, y) {
            return;
        }
        if self.draw_press(x, y) {
            return;
        }
        if self.football_pick_at(x, y) {
            return;
        }
        if let Some(clip) = self.paint_target() {
            let point = self.stage_to_source(&clip, f64::from(x), f64::from(y));
            self.gesture = Gesture::Paint {
                clip: clip.id.clone(),
                tool: BRUSHES[self.brush.min(BRUSHES.len() - 1)],
                size: self.brush_size,
                at: self.source_at_playhead(&clip),
                points: vec![point],
                screen: vec![(x, y)],
            };
            return;
        }
        let (x, y) = (f64::from(x), f64::from(y));
        // As on the lanes: what the inspector holds for the current
        // selection lands before the selection changes under it - and a
        // press on the floor, which clears it, is the commonest way out of
        // a title's text field.
        self.flush_commit();
        let Some(id) = self.stage_hit(x, y) else {
            if !additive {
                self.selection.clear();
                self.transition_selected = None;
            }
            self.gesture = Gesture::None;
            return;
        };
        let already = self.selection.iter().any(|held| held == &id);
        if additive {
            if already {
                self.selection.retain(|held| held != &id);
            } else {
                self.selection.push(id.clone());
            }
        } else if !already {
            self.selection = vec![id.clone()];
        }
        if !self.selection.iter().any(|held| held == &id) {
            self.gesture = Gesture::None;
            return;
        }
        let origins: Vec<StageOrigin> = self
            .stage_clips()
            .into_iter()
            .filter(|clip| {
                self.selection.iter().any(|held| held == &clip.id) && !self.locked(&clip.track_id)
            })
            .map(|clip| {
                // A keyed clip is dragged from where its ride has it at
                // the playhead, not from a constant nothing shows.
                let at = place_in(clip, self.playhead);
                StageOrigin {
                    clip: clip.id.clone(),
                    offset_x: shown(clip, model::KeyProperty::OffsetX, at),
                    offset_y: shown(clip, model::KeyProperty::OffsetY, at),
                }
            })
            .collect();
        self.begin_echo();
        self.stage_guides.clear();
        self.gesture = Gesture::StageMove {
            primary: id,
            origins,
            from: (x, y),
        };
    }

    /// Where a picture's bounds pull to on one axis. Every candidate is a
    /// feature of the moving picture — an edge or the centre — against a
    /// target — the frame's edges and centre, and every other picture's —
    /// and the nearest pair inside `pull` wins. Returns the shift that lands
    /// it and the target's position, for the guide.
    fn stage_snap(features: [f64; 3], targets: &[f64], pull: f64) -> Option<(f64, f64)> {
        let mut best: Option<(f64, f64)> = None;
        for feature in features {
            for &target in targets {
                let shift = target - feature;
                if shift.abs() < pull && best.is_none_or(|(held, _)| shift.abs() < held.abs()) {
                    best = Some((shift, target));
                }
            }
        }
        best
    }

    /// A press on a grip of `id`'s box: a corner scales, the handle above
    /// turns. Both work about the picture's centre, in frame pixels.
    pub fn stage_grip_pressed(&mut self, id: &str, grip: i32, x: f32, y: f32) {
        let Some(clip) = self.clip(id).cloned() else {
            return;
        };
        if self.locked(&clip.track_id) {
            return;
        }
        let footprint = self.footprint(&clip);
        let (width, height) = self.output_size();
        let centre = (
            footprint.cx * f64::from(width),
            footprint.cy * f64::from(height),
        );
        let dx = f64::from(x) * f64::from(width) - centre.0;
        let dy = f64::from(y) * f64::from(height) - centre.1;
        let half = footprint.half_bounds((width, height));
        self.flush_commit();
        self.begin_echo();
        self.gesture = if grip == 4 {
            Gesture::StageRotate {
                clip: id.to_owned(),
                rotation: shown(
                    &clip,
                    model::KeyProperty::Rotation,
                    place_in(&clip, self.playhead),
                ),
                centre,
                from: dy.atan2(dx),
            }
        } else if (grip == 6 || grip == 8) && clip.kind == model::ClipKind::Text {
            // A title's side grips set where its words wrap, not how wide
            // its glyphs are; see `Gesture::TextWidth`.
            Gesture::TextWidth {
                clip: id.to_owned(),
                centre,
                rotation: clip.rotation,
            }
        } else if (grip == 5 || grip == 7) && clip.kind == model::ClipKind::Text {
            // And the top and bottom grips size its box, never its glyphs.
            // https://github.com/jub0t/Concat/issues/119
            Gesture::TextHeight {
                clip: id.to_owned(),
                centre,
                rotation: clip.rotation,
            }
        } else if grip >= 5 {
            // 5 top, 6 right, 7 bottom, 8 left: the pointer's offset from
            // the centre, turned back into the box's own frame.
            let across = grip == 6 || grip == 8;
            let (sin, cos) = clip.rotation.to_radians().sin_cos();
            let along = if across {
                dx * cos + dy * sin
            } else {
                -dx * sin + dy * cos
            };
            Gesture::StageStretch {
                clip: id.to_owned(),
                across,
                stretch: if across {
                    clip.stretch_x
                } else {
                    clip.stretch_y
                },
                centre,
                from: along.abs().max(1.0),
                rotation: clip.rotation,
            }
        } else {
            Gesture::StageScale {
                clip: id.to_owned(),
                scale: shown(
                    &clip,
                    model::KeyProperty::Scale,
                    place_in(&clip, self.playhead),
                ),
                centre,
                from: dx.hypot(dy).max(1.0),
                half,
            }
        };
    }

    /// The pointer moved with a stage gesture live: the echo follows, and
    /// the monitor composites it.
    pub fn stage_dragged(&mut self, x: f32, y: f32, snap: bool) {
        let mut gesture = std::mem::replace(&mut self.gesture, Gesture::None);
        if let Gesture::Draw {
            anchors, screen, ..
        } = &mut gesture
        {
            if let (Some(end), Some(last)) = (anchors.last_mut(), screen.last_mut()) {
                *last = (x, y);
                end.track = None;
            }
            self.gesture = gesture;
            self.request_preview();
            return;
        }
        if let Gesture::Paint {
            clip,
            points,
            screen,
            ..
        } = &mut gesture
        {
            if let Some(clip) = self.clip(clip).cloned() {
                points.push(self.stage_to_source(&clip, f64::from(x), f64::from(y)));
                screen.push((x, y));
            }
            self.gesture = gesture;
            return;
        }
        let (x, y) = (f64::from(x), f64::from(y));
        let (width, height) = self.output_size();
        match &gesture {
            Gesture::StageMove {
                primary,
                origins,
                from,
            } => {
                let (mut dx, mut dy) = (x - from.0, y - from.1);
                if snap {
                    // Shift holds the axis the drag is mostly along, measured
                    // in pixels so a tall frame does not bias it.
                    if (dx * f64::from(width)).abs() >= (dy * f64::from(height)).abs() {
                        dy = 0.0;
                    } else {
                        dx = 0.0;
                    }
                }
                self.stage_guides.clear();
                if self.lanes.snap {
                    // The same pull on both axes, in frame pixels: a
                    // hundredth of the long side, which is about eight
                    // pixels on a stage of the size a laptop gives it.
                    let pull = 0.01 * f64::from(width.max(height));
                    let frame = (width, height);
                    let moving = origins
                        .iter()
                        .find(|origin| &origin.clip == primary)
                        .and_then(|origin| self.clip(&origin.clip).map(|clip| (origin, clip)));
                    if let Some((origin, clip)) = moving {
                        let (hw, hh) = self.footprint(clip).half_bounds(frame);
                        let cx = 0.5 + origin.offset_x + dx;
                        let cy = 0.5 + origin.offset_y + dy;
                        // The frame's own lines, then every picture that is
                        // staying put.
                        let mut xs = vec![0.0, 0.5, 1.0];
                        let mut ys = vec![0.0, 0.5, 1.0];
                        for other in self.stage_clips() {
                            if origins.iter().any(|origin| origin.clip == other.id) {
                                continue;
                            }
                            let footprint = self.footprint(other);
                            let (ow, oh) = footprint.half_bounds(frame);
                            xs.extend([footprint.cx - ow, footprint.cx, footprint.cx + ow]);
                            ys.extend([footprint.cy - oh, footprint.cy, footprint.cy + oh]);
                        }
                        if let Some((shift, at)) =
                            Self::stage_snap([cx - hw, cx, cx + hw], &xs, pull / f64::from(width))
                        {
                            dx += shift;
                            self.stage_guides.push(StageGuideData {
                                vertical: true,
                                at: at as f32,
                            });
                        }
                        if let Some((shift, at)) =
                            Self::stage_snap([cy - hh, cy, cy + hh], &ys, pull / f64::from(height))
                        {
                            dy += shift;
                            self.stage_guides.push(StageGuideData {
                                vertical: false,
                                at: at as f32,
                            });
                        }
                    }
                }
                let playhead = self.playhead;
                for origin in origins {
                    if let Some(clip) = self.echo_clip_mut(&origin.clip) {
                        let at = place_in(clip, playhead);
                        let (x, y) = (
                            (origin.offset_x + dx).clamp(-1.0, 1.0),
                            (origin.offset_y + dy).clamp(-1.0, 1.0),
                        );
                        write_keyable(clip, model::KeyProperty::OffsetX, x, at);
                        write_keyable(clip, model::KeyProperty::OffsetY, y, at);
                    }
                }
            }
            Gesture::StageScale {
                clip,
                scale,
                centre,
                from,
                half,
            } => {
                let dx = x * f64::from(width) - centre.0;
                let dy = y * f64::from(height) - centre.1;
                let mut next = scale * dx.hypot(dy) / from;
                if snap {
                    next = (next * 20.0).round() / 20.0;
                }
                self.stage_guides.clear();
                if self.lanes.snap && *scale > 0.0 {
                    // The edges pull to the same lines a move pulls to - the
                    // frame's, and every other picture's - but here the pull
                    // sets the size, not the place: the scale that lands the
                    // nearest edge on its target, when one is inside reach.
                    let pull = 0.01 * f64::from(width.max(height));
                    let frame = (width, height);
                    let (cx, cy) = (centre.0 / f64::from(width), centre.1 / f64::from(height));
                    let mut xs = vec![0.0, 0.5, 1.0];
                    let mut ys = vec![0.0, 0.5, 1.0];
                    for other in self.stage_clips() {
                        if other.id == *clip {
                            continue;
                        }
                        let footprint = self.footprint(other);
                        let (ow, oh) = footprint.half_bounds(frame);
                        xs.extend([footprint.cx - ow, footprint.cx, footprint.cx + ow]);
                        ys.extend([footprint.cy - oh, footprint.cy, footprint.cy + oh]);
                    }
                    // Four edges: each is the centre plus or minus a half
                    // bound that grows with the scale.
                    let edges = [
                        (-1.0, half.0, cx, &xs, true, width),
                        (1.0, half.0, cx, &xs, true, width),
                        (-1.0, half.1, cy, &ys, false, height),
                        (1.0, half.1, cy, &ys, false, height),
                    ];
                    let mut best: Option<(f64, f64, bool, f64)> = None;
                    for (sign, base, at_centre, targets, vertical, extent) in edges {
                        if base <= 0.0 {
                            continue;
                        }
                        let edge = at_centre + sign * base * next / scale;
                        for &target in targets.iter() {
                            let distance = (target - edge).abs() * f64::from(extent);
                            let wanted = (target - at_centre) * sign;
                            if distance < pull
                                && wanted > 0.0
                                && best.is_none_or(|(held, ..)| distance < held)
                            {
                                best = Some((distance, scale * wanted / base, vertical, target));
                            }
                        }
                    }
                    if let Some((_, snapped, vertical, at)) = best {
                        next = snapped;
                        self.stage_guides.push(StageGuideData {
                            vertical,
                            at: at as f32,
                        });
                    }
                }
                let playhead = self.playhead;
                if let Some(clip) = self.echo_clip_mut(clip) {
                    let at = place_in(clip, playhead);
                    write_keyable(clip, model::KeyProperty::Scale, next.clamp(0.05, 8.0), at);
                }
            }
            Gesture::StageStretch {
                clip,
                across,
                stretch,
                centre,
                from,
                rotation,
            } => {
                let dx = x * f64::from(width) - centre.0;
                let dy = y * f64::from(height) - centre.1;
                let (sin, cos) = rotation.to_radians().sin_cos();
                let along = if *across {
                    dx * cos + dy * sin
                } else {
                    -dx * sin + dy * cos
                };
                let mut next = stretch * along.abs() / from;
                if snap {
                    next = (next * 20.0).round() / 20.0;
                }
                if let Some(clip) = self.echo_clip_mut(clip) {
                    if *across {
                        clip.stretch_x = next.clamp(0.1, 10.0);
                    } else {
                        clip.stretch_y = next.clamp(0.1, 10.0);
                    }
                }
            }
            Gesture::TextWidth {
                clip,
                centre,
                rotation,
            } => {
                let dx = x * f64::from(width) - centre.0;
                let dy = y * f64::from(height) - centre.1;
                let (sin, cos) = rotation.to_radians().sin_cos();
                let along = (dx * cos + dy * sin).abs();
                // Twice the reach, as a fraction of the frame; no narrower
                // than a few percent, or every word stands alone.
                let mut next = (2.0 * along / f64::from(width)).clamp(0.03, 2.0);
                if snap {
                    next = (next * 20.0).round() / 20.0;
                }
                if let Some(clip) = self.echo_clip_mut(clip) {
                    clip.text.get_or_insert_with(TextStyle::default).max_width = next;
                }
            }
            Gesture::TextHeight {
                clip,
                centre,
                rotation,
            } => {
                let dx = x * f64::from(width) - centre.0;
                let dy = y * f64::from(height) - centre.1;
                let (sin, cos) = rotation.to_radians().sin_cos();
                // The pointer's reach across the box's own axis.
                let across = (-dx * sin + dy * cos).abs();
                let mut next = (2.0 * across / f64::from(height)).clamp(0.03, 2.0);
                if snap {
                    next = (next * 20.0).round() / 20.0;
                }
                if let Some(clip) = self.echo_clip_mut(clip) {
                    clip.text.get_or_insert_with(TextStyle::default).max_height = next;
                }
            }
            Gesture::StageRotate {
                clip,
                rotation,
                centre,
                from,
            } => {
                let dx = x * f64::from(width) - centre.0;
                let dy = y * f64::from(height) - centre.1;
                let swept = (dy.atan2(dx) - from).to_degrees();
                let mut next = rotation + swept;
                if snap {
                    next = (next / 15.0).round() * 15.0;
                }
                // Kept in the inspector's range, wrapping rather than
                // stopping: a turn through the bottom carries on.
                next = (next + 180.0).rem_euclid(360.0) - 180.0;
                let playhead = self.playhead;
                if let Some(clip) = self.echo_clip_mut(clip) {
                    let at = place_in(clip, playhead);
                    write_keyable(clip, model::KeyProperty::Rotation, next, at);
                }
            }
            _ => {
                self.gesture = gesture;
                return;
            }
        }
        self.gesture = gesture;
        self.request_preview();
    }

    /// The stage gesture is over: whatever the echo moved becomes one
    /// transform command per picture, batched when there are several.
    pub fn stage_released(&mut self) {
        self.stage_guides.clear();
        if matches!(self.gesture, Gesture::Draw { .. }) {
            self.draw_released();
            return;
        }
        let overlay = matches!(self.gesture, Gesture::Paint { .. })
            .then(|| self.stroke_overlay())
            .filter(|(path, _, _)| !path.is_empty());
        let gesture = std::mem::replace(&mut self.gesture, Gesture::None);
        let touched: Vec<String> = match gesture {
            Gesture::StageMove { origins, .. } => {
                origins.into_iter().map(|origin| origin.clip).collect()
            }
            Gesture::StageScale { clip, .. }
            | Gesture::StageRotate { clip, .. }
            | Gesture::StageStretch { clip, .. }
            | Gesture::TextWidth { clip, .. } => vec![clip],
            Gesture::Paint {
                clip,
                tool,
                size,
                at,
                points,
                ..
            } => {
                // The stroke becomes one command, and one undo step; a
                // smart stroke then has its thing read from the frame, and
                // stays drawn until it has been.
                let stroke = model::Stroke {
                    tool,
                    size,
                    points: points.clone(),
                    at: Some(at),
                };
                self.pending_stroke = stroke.is_smart().then_some(overlay).flatten();
                self.apply(Command::AddCutoutStroke {
                    clip_id: clip,
                    stroke,
                });
                self.ensure_regions();
                return;
            }
            other => {
                // Not ours to end; a lane gesture is still live.
                self.gesture = other;
                return;
            }
        };
        let Some(echo) = self.echo.take() else {
            return;
        };
        let mut commands = Vec::new();
        for id in touched {
            let (Some(after), Some(before)) = (
                echo.active().clip(&id),
                self.session
                    .as_ref()
                    .and_then(|session| session.project().active().clip(&id)),
            ) else {
                continue;
            };
            if after.scale != before.scale
                || after.offset_x != before.offset_x
                || after.offset_y != before.offset_y
                || after.rotation != before.rotation
                || after.stretch_x != before.stretch_x
                || after.stretch_y != before.stretch_y
            {
                commands.push(Command::SetClipTransform {
                    clip_id: id.clone(),
                    scale: Some(after.scale),
                    offset_x: Some(after.offset_x),
                    offset_y: Some(after.offset_y),
                    rotation: Some(after.rotation),
                    stretch_x: Some(after.stretch_x),
                    stretch_y: Some(after.stretch_y),
                });
            }
            // A keyed property is dragged as a key at the playhead, not as
            // its constant - see `write_keyable` - so the keys the gesture
            // put on the echo have to be committed too, or the picture
            // snaps back to the ride the document still holds.
            // https://github.com/jub0t/Concat/issues/228
            commands.extend(key_commands(&id, before, after));
            // A title's wrap width, from its side grips.
            if after.text != before.text {
                commands.push(Command::UpdateClip {
                    clip_id: id,
                    patch: ClipPatch {
                        text: Some(after.text.clone()),
                        ..Default::default()
                    },
                });
            }
        }
        match commands.len() {
            0 => {
                // A click that moved nothing: the echo is gone and the
                // monitor goes back to the document.
                self.request_preview();
            }
            1 => {
                self.apply(commands.remove(0));
            }
            _ => {
                self.apply(Command::Batch { commands });
            }
        }
    }

    // ── cutouts ──
    //
    // A cutout is a mask per source instant, found by the host's model and
    // cached in the project folder. The window's part is to notice which
    // media need masks they do not have, run one analysis at a time, and
    // turn a brush on the stage into strokes on the document.

    /// Starts the analysis for the first media whose cutout wants masks
    /// that are not there, unless one is running. Called after every
    /// change; the finished job calls it again for whatever is next.
    pub fn ensure_cutouts(&mut self) {
        if !self.cutout_jobs.is_empty() || self.host.cutouts.is_busy() {
            return;
        }
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let project = std::path::PathBuf::from(session.path());
        // One analysis per media and subject, keyed the way the job map is.
        let wanted: Vec<(String, AnalyseRequest)> = Cutouts::requests(self.project(), &project)
            .into_iter()
            .map(|(media_id, request)| (Self::analysis_key(&media_id, request.subject), request))
            .collect();
        let Some((id, request)) = wanted
            .into_iter()
            .find(|(_, request)| Cutouts::outstanding(request) > 0)
        else {
            return;
        };
        self.cutout_jobs.insert(id.clone(), (false, 0.0));
        let cutouts = Arc::clone(&self.host.cutouts);
        let epoch = crate::host::project_epoch();
        spawn_in_project(
            move || {
                let mut last = (false, -1.0f32);
                let reporting = id.clone();
                let result = cutouts.analyse(&request, &mut |progress| {
                    // Every percent, not every frame: the readout cannot
                    // use more and the event loop has other work.
                    let now = match progress {
                        concat_host::cutout::Progress::Fetching { received, total } => {
                            (true, received as f32 / total.max(1) as f32)
                        }
                        concat_host::cutout::Progress::Analysing(fraction) => (false, fraction),
                    };
                    if now.0 != last.0 || now.1 - last.1 >= 0.01 {
                        last = now;
                        let id = reporting.clone();
                        on_ui_in_project(epoch, move |studio, _, _| {
                            if let Some(held) = studio.cutout_jobs.get_mut(&id) {
                                *held = now;
                            }
                        });
                    }
                });
                (id, result)
            },
            |studio, _, _, (id, result)| {
                studio.cutout_jobs.remove(&id);
                match result {
                    Ok(_) => {
                        // The monitor shows the cut, and whatever else is
                        // waiting gets its turn.
                        studio.request_preview();
                        studio.ensure_cutouts();
                    }
                    Err(error) if error.contains("cancelled") => {}
                    Err(error) => studio.notify(&tf("studio.removeBackground", &[&error]), true),
                }
            },
        );
    }

    // ── drawing ──
    //
    // Telestration, the way KlipDraw does it: lines, arrows, curved runs,
    // zones and lines joining players, drawn on the monitor over the
    // selected clip and kept on it as drawings that appear at the playhead,
    // draw themselves on and fade after their hold. A point placed on a
    // tracked player keeps to his feet, so a line from a defender follows
    // him. Line, arrow and curve are a drag; a zone and a link are a click
    // per point and Finish.

    /// The drawing kinds, in the inspector's order.
    const DRAW_KINDS: [model::DrawKind; 5] = [
        model::DrawKind::Line,
        model::DrawKind::Arrow,
        model::DrawKind::Curve,
        model::DrawKind::Zone,
        model::DrawKind::Link,
    ];

    /// Picks a drawing tool by its row, or puts it away when it is the one
    /// already out: the stage draws while one is out.
    pub fn draw_tool(&mut self, index: i32) {
        let wanted = usize::try_from(index)
            .ok()
            .filter(|index| *index < Self::DRAW_KINDS.len());
        self.draw_tool = if wanted == self.draw_tool {
            None
        } else {
            wanted
        };
        self.draw_draft = None;
        if self.draw_tool.is_some() {
            self.painting = false;
            self.football_picking = false;
        }
        self.request_preview();
    }

    /// One of the drawing tool's settings: 0 dashed (0 or 1), 1 thickness,
    /// 2 hold in seconds, 3 reveal in seconds, 4 bend. They shape the next
    /// drawing, and the selected clip's last one too, so a line just drawn
    /// can be adjusted without drawing it again.
    pub fn draw_option(&mut self, which: i32, value: f32) {
        let value = f64::from(value);
        match which {
            0 => self.draw_style.dashed = value >= 0.5,
            1 => self.draw_style.thickness = value.clamp(model::MARK_SCALE.0, model::MARK_SCALE.1),
            2 => self.draw_style.hold = value.clamp(0.1, 600.0),
            3 => self.draw_style.reveal = value.clamp(0.0, 10.0),
            5 => {
                // The freeze's own length, not the last drawing's.
                self.draw_style.freeze = value.clamp(0.5, 60.0);
                return;
            }
            _ => self.draw_style.bend = value.clamp(-1.0, 1.0),
        }
        self.restyle_last_drawing(&format!("draw-option-{which}"));
    }

    /// The drawing colour, as a row of the marks' palette.
    pub fn draw_colour(&mut self, index: i32) {
        if let Some(colour) = usize::try_from(index)
            .ok()
            .and_then(|index| Self::MARK_COLOURS.get(index).copied())
        {
            self.draw_style.colour = colour;
            self.restyle_last_drawing("draw-colour");
        }
    }

    /// The selected clip's last drawing given the tool's current style.
    fn restyle_last_drawing(&mut self, gesture: &str) {
        let Some(id) = self.sole_selection() else {
            return;
        };
        let Some(clip) = self.clip(&id).cloned() else {
            return;
        };
        let mut drawings = clip.drawings.clone();
        let Some(last) = drawings.last_mut() else {
            return;
        };
        let style = &self.draw_style;
        last.dashed = style.dashed;
        last.thickness = style.thickness;
        last.hold = style.hold;
        last.reveal = style.reveal.min(style.hold);
        last.bend = style.bend;
        last.colour = style.colour;
        if drawings != clip.drawings {
            self.apply_in(
                Some(gesture),
                Command::SetClipDrawings {
                    clip_id: id,
                    drawings,
                },
            );
        }
    }

    /// The clip the drawing tool draws on: the one selected, if it is a
    /// picture showing on the stage.
    fn draw_target(&self) -> Option<Clip> {
        self.draw_tool?;
        let clip = self
            .sole_selection()
            .and_then(|id| self.clip(&id).cloned())?;
        (clip.kind.is_visual()
            && !self.locked(&clip.track_id)
            && self
                .stage_clips()
                .into_iter()
                .any(|shown| shown.id == clip.id))
        .then_some(clip)
    }

    /// The point a stage press lands on, as a drawing keeps it: on the
    /// feet of the tracked player under it when there is one.
    fn anchor_at(&self, clip: &Clip, x: f32, y: f32) -> model::Anchor {
        let at = self.stage_to_source(clip, f64::from(x), f64::from(y));
        let tracked = self
            .football_request(&clip.id)
            .and_then(|request| concat_vision::football::shared(&request.dir()))
            .and_then(|set| set.pick(self.source_at_playhead(clip), at[0] as f32, at[1] as f32));
        match tracked {
            Some(tracked) => {
                let [l, _, r, b] = tracked.bbox;
                model::Anchor {
                    at: [f64::from((l + r) * 0.5), f64::from(b)],
                    track: Some(tracked.id),
                }
            }
            None => model::Anchor { at, track: None },
        }
    }

    /// A press on the stage while a drawing tool is out: the start of a
    /// line, or the next point of a zone or a link. `false` when no tool
    /// is out or nothing is there to draw on.
    fn draw_press(&mut self, x: f32, y: f32) -> bool {
        let Some(clip) = self.draw_target() else {
            return false;
        };
        let kind = Self::DRAW_KINDS[self.draw_tool.unwrap_or(0)];
        let anchor = self.anchor_at(&clip, x, y);
        match kind {
            model::DrawKind::Zone | model::DrawKind::Link => {
                let draft = self
                    .draw_draft
                    .take()
                    .filter(|(held, _, _)| *held == clip.id)
                    .unwrap_or((clip.id.clone(), Vec::new(), Vec::new()));
                let (id, mut anchors, mut screen) = draft;
                anchors.push(anchor);
                screen.push((x, y));
                self.draw_draft = Some((id, anchors, screen));
            }
            _ => {
                self.gesture = Gesture::Draw {
                    clip: clip.id.clone(),
                    anchors: vec![anchor.clone(), anchor],
                    screen: vec![(x, y), (x, y)],
                };
            }
        }
        self.request_preview();
        true
    }

    /// The drawing in flight, as the stage's overlay draws it.
    fn draw_overlay(&self) -> Option<(String, f32, bool)> {
        let (screen, closed) = match (&self.gesture, &self.draw_draft) {
            (Gesture::Draw { screen, .. }, _) => (screen, false),
            (_, Some((_, _, screen))) => (
                screen,
                self.draw_tool.map(|index| Self::DRAW_KINDS[index]) == Some(model::DrawKind::Zone),
            ),
            _ => return None,
        };
        let (first, rest) = screen.split_first()?;
        let mut path = format!("M {:.1} {:.1}", first.0 * 1000.0, first.1 * 1000.0);
        for (x, y) in rest.iter().chain(rest.is_empty().then_some(first)) {
            path.push_str(&format!(" L {:.1} {:.1}", x * 1000.0, y * 1000.0));
        }
        if closed && screen.len() > 2 {
            path.push_str(" Z");
        }
        Some((path, 0.004 * self.draw_style.thickness as f32, false))
    }

    /// A drawing of `kind` through `anchors` on clip `id`, starting at the
    /// playhead in the tool's style, as one undo step.
    fn add_drawing(&mut self, id: &str, kind: model::DrawKind, points: Vec<model::Anchor>) {
        let Some(clip) = self.clip(id).cloned() else {
            return;
        };
        let style = self.draw_style.clone();
        let mut drawings = clip.drawings.clone();
        let here = (f64::from(self.playhead) - clip.start).clamp(0.0, clip.duration);
        // On a still - a frozen frame - the drawings come one after another,
        // each when the last has drawn itself on, and stay to the end: the
        // way a broadcast walks through a frozen play. On footage each one
        // is where the playhead is, for its hold.
        let (start, hold) = if clip.kind == model::ClipKind::Image {
            let after = drawings
                .iter()
                .map(|drawing| drawing.start + drawing.reveal)
                .fold(here, f64::max)
                .min((clip.duration - 0.1).max(0.0));
            (after, (clip.duration - after).max(0.1))
        } else {
            (here, style.hold)
        };
        drawings.push(model::Drawing {
            kind,
            points,
            dashed: style.dashed,
            colour: style.colour,
            thickness: style.thickness,
            bend: style.bend,
            start,
            hold,
            reveal: style.reveal.min(hold),
        });
        self.apply(Command::SetClipDrawings {
            clip_id: id.to_owned(),
            drawings,
        });
    }

    /// The end of a drag with a line, an arrow or a curve: the drawing,
    /// unless the drag was too short to be one.
    fn draw_released(&mut self) {
        let gesture = std::mem::replace(&mut self.gesture, Gesture::None);
        let Gesture::Draw {
            clip,
            mut anchors,
            screen,
        } = gesture
        else {
            self.gesture = gesture;
            return;
        };
        let (Some(first), Some(last)) = (screen.first(), screen.last()) else {
            return;
        };
        if (last.0 - first.0).hypot(last.1 - first.1) < 0.01 {
            self.request_preview();
            return;
        }
        // The end lands on whoever is under it, now the pointer has stopped.
        if let Some(target) = self.clip(&clip).cloned()
            && let Some(end) = anchors.last_mut()
        {
            *end = self.anchor_at(&target, last.0, last.1);
        }
        let kind = Self::DRAW_KINDS[self.draw_tool.unwrap_or(0)];
        self.add_drawing(&clip, kind, anchors);
    }

    /// Freeze and draw: the picture under the playhead held for the
    /// freeze's length, the still selected and the playhead on it, ready
    /// for the drawings to be laid out one after another.
    pub fn draw_freeze(&mut self) {
        let Some(id) = self.freeze_for(self.draw_style.freeze) else {
            return;
        };
        if let Some(still) = self.clip(&id).cloned() {
            self.seek(still.start as f32);
        }
        // The arrow out, whatever tool was: a player tool left out would
        // take the presses meant for the drawings.
        if !(10..=14).contains(&self.analysis_tool) {
            self.analysis_tool_pick(11);
        }
        self.painting = false;
        self.football_picking = false;
        self.notify(&t("draw.frozen"), false);
    }

    /// Finish: the zone or the link being clicked out becomes a drawing.
    pub fn draw_finish(&mut self) {
        let Some((id, anchors, _)) = self.draw_draft.take() else {
            return;
        };
        let kind = Self::DRAW_KINDS[self.draw_tool.unwrap_or(3)];
        let least = if kind == model::DrawKind::Zone { 3 } else { 2 };
        if anchors.len() < least {
            self.notify(&t("draw.morePoints"), true);
            return;
        }
        self.add_drawing(&id, kind, anchors);
    }

    /// Takes the selected clip's last drawing away, or the point last
    /// clicked when a zone or a link is being drawn.
    pub fn draw_undo(&mut self) {
        if let Some((_, anchors, screen)) = self.draw_draft.as_mut() {
            anchors.pop();
            screen.pop();
            if anchors.is_empty() {
                self.draw_draft = None;
            }
            self.request_preview();
            return;
        }
        let Some(id) = self.sole_selection() else {
            return;
        };
        let Some(clip) = self.clip(&id).cloned() else {
            return;
        };
        if clip.drawings.is_empty() {
            return;
        }
        let mut drawings = clip.drawings.clone();
        drawings.pop();
        self.apply(Command::SetClipDrawings {
            clip_id: id,
            drawings,
        });
    }

    /// Takes every drawing off the selected clip.
    pub fn draw_clear(&mut self) {
        self.draw_draft = None;
        let Some(id) = self.sole_selection() else {
            return;
        };
        if self.clip(&id).is_some_and(|clip| !clip.drawings.is_empty()) {
            self.apply(Command::SetClipDrawings {
                clip_id: id,
                drawings: Vec::new(),
            });
        }
    }

    // ── the pitch board ──
    //
    // The campograma: the players and the ball as dots on a pitch, live
    // with the playhead, in their teams' kit colours - the board every
    // analysis tool keeps beside the picture. The picture becomes metres
    // through a calibration: a few points clicked on the video, each then
    // on the board, where it snaps to the landmark it is; the camera's
    // motion, measured from the tracks, carries it along the clip.

    /// The picture whose players the board shows: the one selected if it
    /// is footage, else the first footage on the stage.
    fn board_clip(&self) -> Option<Clip> {
        let picked = self
            .sole_selection()
            .and_then(|id| self.clip(&id).cloned())
            .filter(|clip| {
                clip.kind == model::ClipKind::Video || clip.kind == model::ClipKind::Image
            });
        picked.or_else(|| {
            self.stage_clips()
                .into_iter()
                .find(|clip| clip.kind == model::ClipKind::Video)
                .cloned()
        })
    }

    /// The camera's motion through clip `id`'s analysis and its
    /// calibration, read once per change of either file.
    fn board_state(
        &self,
        id: &str,
    ) -> Option<(
        Arc<concat_vision::football::TrackSet>,
        Arc<concat_vision::football::pitch::Motion>,
        concat_vision::football::pitch::Calibration,
        Arc<HashMap<u32, (i32, String, i32)>>,
    )> {
        let request = self.football_request(id)?;
        let dir = request.dir();
        let set = concat_vision::football::shared(&dir)?;
        let stamp = |name: &str| {
            std::fs::metadata(dir.join(name))
                .and_then(|meta| meta.modified())
                .ok()
        };
        let key = (
            dir.clone(),
            stamp("tracks.json"),
            stamp("calibration.json"),
            stamp("roster.json"),
        );
        let mut cache = self.board_cache.borrow_mut();
        if let Some((held, motion, calibration, numbers)) = cache.as_ref()
            && *held == key
        {
            return Some((
                set,
                Arc::clone(motion),
                calibration.clone(),
                Arc::clone(numbers),
            ));
        }
        let motion = Arc::new(concat_vision::football::pitch::Motion::of(&set));
        let calibration = concat_vision::football::pitch::Calibration::load(&dir);
        // Each team's numbers: its players by id, 1 upward, so a number
        // holds for as long as the analysis does. One pass over the match.
        let mut teams: HashMap<u32, u8> = HashMap::new();
        for tracked in set.samples.iter().flat_map(|sample| sample.boxes.iter()) {
            if tracked.kind == concat_vision::football::Kind::Player && tracked.team > 0 {
                teams.insert(tracked.id, tracked.team);
            }
        }
        let mut numbers: HashMap<u32, (i32, String, i32)> = HashMap::new();
        for team in [1u8, 2] {
            let mut ids: Vec<u32> = teams
                .iter()
                .filter(|(_, held)| **held == team)
                .map(|(id, _)| *id)
                .collect();
            ids.sort_unstable();
            for (index, id) in ids.into_iter().enumerate() {
                numbers.insert(id, (index as i32 + 1, String::new(), 0));
            }
        }
        // Over them, the real numbers, names and roles the analyst gave,
        // handed on to the tracks that pick a lost player up.
        for (id, player) in concat_vision::football::Roster::load(&dir).effective(&set) {
            let entry = numbers.entry(id).or_insert((0, String::new(), 0));
            if let Some(number) = player.number {
                entry.0 = number as i32;
            }
            entry.1 = player.name.trim().to_string();
            entry.2 = player
                .role
                .and_then(|role| {
                    concat_vision::football::Role::ALL
                        .iter()
                        .position(|r| *r == role)
                })
                .map_or(0, |index| index as i32 + 1);
        }
        let numbers = Arc::new(numbers);
        *cache = Some((
            key,
            Arc::clone(&motion),
            calibration.clone(),
            Arc::clone(&numbers),
        ));
        Some((set, motion, calibration, numbers))
    }

    /// The board's dots at the playhead, as fractions of the pitch.
    /// The board's dots and its team lines at the playhead: each player
    /// with his number in his team, the ink that reads on his kit and the
    /// last three seconds of his run; and for each team its back line and
    /// its midfield line, which a coach reads the block's height and width
    /// from.
    fn board_layers(&self) -> (Vec<PitchDotData>, Vec<PitchLineData>) {
        use concat_vision::football::{Kind, pitch};
        let Some(clip) = self.board_clip() else {
            return (Vec::new(), Vec::new());
        };
        let Some((set, motion, calibration, numbers)) = self.board_state(&clip.id) else {
            return (Vec::new(), Vec::new());
        };
        let seconds = self.source_at_playhead(&clip);
        let to_ms = |seconds: f64| (seconds.max(0.0) * 1000.0).round() as u64;
        let Some(map) = pitch::map_at(&calibration, &motion, to_ms(seconds)) else {
            return (Vec::new(), Vec::new());
        };
        let kit_of = |team: u8| -> [u8; 3] {
            set.team_colours
                .get(usize::from(team.max(1)) - 1)
                .copied()
                .unwrap_or(if team == 2 {
                    [91, 141, 239]
                } else {
                    [242, 201, 76]
                })
        };
        // The trail's instants, each with its map, worked out once for
        // every player.
        let trail_maps: Vec<(f64, concat_vision::football::pitch::Matrix)> = (0..=7)
            .rev()
            .map(|step| seconds - f64::from(step) * 0.4)
            .filter(|at| *at >= 0.0)
            .filter_map(|at| pitch::map_at(&calibration, &motion, to_ms(at)).map(|map| (at, map)))
            .collect();
        let place = |map: &concat_vision::football::pitch::Matrix,
                     tracked: &concat_vision::football::TrackedBox|
         -> Option<(f64, f64)> {
            let point = if tracked.kind == Kind::Ball {
                let (x, y) = concat_vision::football::centre(&tracked.bbox);
                (f64::from(x), f64::from(y))
            } else {
                pitch::feet(&tracked.bbox)
            };
            let (x, y) = pitch::apply(map, point)?;
            ((-5.0..=pitch::LENGTH + 5.0).contains(&x) && (-5.0..=pitch::WIDTH + 5.0).contains(&y))
                .then_some((x, y))
        };
        let to_board = |(x, y): (f64, f64)| ((x / pitch::LENGTH) as f32, (y / pitch::WIDTH) as f32);
        // The run behind each player: where he was each 0.4 s of the last
        // three, on the board, as path commands over a 1000 square.
        let trail = |id: u32| -> String {
            let mut points = Vec::new();
            for (at, map) in &trail_maps {
                if let Some(tracked) = set.track_at(id, *at)
                    && let Some(point) = place(map, &tracked)
                {
                    let (bx, by) = to_board(point);
                    points.push((bx * 1000.0, by * 1000.0));
                }
            }
            if points.len() < 2 {
                return String::new();
            }
            let mut path = format!("M {:.1} {:.1}", points[0].0, points[0].1);
            for (x, y) in &points[1..] {
                path.push_str(&format!(" L {x:.1} {y:.1}"));
            }
            path
        };
        let mut dots = Vec::new();
        // Each team's outfield players on the board, for its lines.
        let mut outfield: HashMap<u8, Vec<(f64, f64)>> = HashMap::new();
        for tracked in set.at(seconds) {
            let Some(point) = place(&map, &tracked) else {
                continue;
            };
            let (x, y) = to_board(point);
            let (kind, colour) = match tracked.kind {
                Kind::Ball => (3, [255, 255, 255]),
                Kind::Referee => (2, [52, 58, 64]),
                Kind::Goalkeeper => (1, [155, 89, 182]),
                Kind::Player if tracked.team > 0 => (0, kit_of(tracked.team)),
                Kind::Player => (0, [139, 148, 158]),
            };
            if tracked.kind == Kind::Player && tracked.team > 0 {
                outfield.entry(tracked.team).or_default().push(point);
            }
            let [r, g, b] = colour;
            let light = 0.3 * f32::from(r) + 0.59 * f32::from(g) + 0.11 * f32::from(b) > 150.0;
            dots.push(PitchDotData {
                x,
                y,
                colour: slint::Color::from_rgb_u8(r, g, b),
                ink: if light {
                    slint::Color::from_rgb_u8(14, 17, 22)
                } else {
                    slint::Color::from_rgb_u8(255, 255, 255)
                },
                kind,
                id: tracked.id as i32,
                number: numbers.get(&tracked.id).map_or(0, |held| held.0),
                name: numbers
                    .get(&tracked.id)
                    .map(|held| SharedString::from(held.1.as_str()))
                    .unwrap_or_default(),
                chosen: self.football_chosen == Some(tracked.id),
                trail: if tracked.kind == Kind::Ball {
                    SharedString::new()
                } else {
                    trail(tracked.id).into()
                },
            });
        }
        // The lines: a team defends the goal its players stand nearer; its
        // four deepest outfield players are the back line, the next four
        // the midfield, each drawn across the pitch.
        let mut lines = Vec::new();
        for (team, players) in &outfield {
            if players.len() < 6 {
                continue;
            }
            let mean_x = players.iter().map(|p| p.0).sum::<f64>() / players.len() as f64;
            let own_goal = if mean_x < pitch::LENGTH / 2.0 {
                0.0
            } else {
                pitch::LENGTH
            };
            let mut by_depth = players.clone();
            by_depth.sort_by(|a, b| (a.0 - own_goal).abs().total_cmp(&(b.0 - own_goal).abs()));
            let [r, g, b] = kit_of(*team);
            for line in [&by_depth[..4], &by_depth[4..by_depth.len().min(8)]] {
                if line.len() < 3 {
                    continue;
                }
                let mut line = line.to_vec();
                line.sort_by(|a, b| a.1.total_cmp(&b.1));
                let mut path = String::new();
                for (index, point) in line.iter().enumerate() {
                    let (bx, by) = to_board(*point);
                    path.push_str(&format!(
                        "{} {:.1} {:.1} ",
                        if index == 0 { "M" } else { "L" },
                        bx * 1000.0,
                        by * 1000.0
                    ));
                }
                lines.push(PitchLineData {
                    commands: path.trim_end().into(),
                    colour: slint::Color::from_rgb_u8(r, g, b),
                });
            }
        }
        (dots, lines)
    }

    /// The board's legend: each team's kit colour.
    fn board_kits(set: &concat_vision::football::TrackSet) -> Vec<slint::Color> {
        set.team_colours
            .iter()
            .map(|[r, g, b]| slint::Color::from_rgb_u8(*r, *g, *b))
            .collect()
    }

    /// The calibration being made, its points on the board.
    fn board_pairs(&self) -> Vec<PitchDotData> {
        use concat_vision::football::pitch;
        self.calibrating
            .as_ref()
            .map(|(_, _, pairs, _)| {
                pairs
                    .iter()
                    .map(|pair| PitchDotData {
                        x: (pair[2] / pitch::LENGTH) as f32,
                        y: (pair[3] / pitch::WIDTH) as f32,
                        colour: slint::Color::from_rgb_u8(0x2d, 0xd4, 0xbf),
                        ink: slint::Color::from_rgb_u8(0x0e, 0x11, 0x16),
                        kind: 4,
                        id: -1,
                        number: 0,
                        chosen: false,
                        trail: SharedString::new(),
                        name: SharedString::new(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Calibration started, or put away unsaved: the next stage click is a
    /// point of the video, the next board click the landmark it shows.
    pub fn pitch_calibrate(&mut self, on: bool) {
        if !on {
            self.calibrating = None;
            if self.analysis_tool == 20 {
                self.analysis_tool = 0;
            }
            return;
        }
        let Some(clip) = self.board_clip() else {
            self.notify(&t("pitch.needsFootage"), true);
            return;
        };
        if !self.football_analysed(&clip.id) {
            // The board needs the players; the analysis runs, and the
            // calibration can be made while it does.
            self.selection = vec![clip.id.clone()];
            if self.football_job.is_none() {
                self.football_analyse();
            }
        }
        let ms = (self.source_at_playhead(&clip).max(0.0) * 1000.0).round() as u64;
        self.calibrating = Some((clip.id.clone(), ms, Vec::new(), None));
        self.draw_tool = None;
        self.football_picking = false;
        self.analysis_tool = 20;
        self.pause();
    }

    /// A stage press while calibrating: the video's half of the next pair.
    fn calibrate_press(&mut self, x: f32, y: f32) -> bool {
        let Some((id, _, _, _)) = self.calibrating.clone() else {
            return false;
        };
        let Some(clip) = self.clip(&id).cloned() else {
            return true;
        };
        let [u, v] = self.stage_to_source(&clip, f64::from(x), f64::from(y));
        if let Some(state) = self.calibrating.as_mut() {
            state.3 = Some((u, v));
        }
        self.request_preview();
        true
    }

    /// A click on the board: while calibrating with a video point waiting,
    /// its landmark - snapped to the nearest within four metres.
    pub fn pitch_clicked(&mut self, fx: f32, fy: f32) {
        use concat_vision::football::pitch;
        let Some(state) = self.calibrating.as_mut() else {
            return;
        };
        let Some((u, v)) = state.3.take() else {
            self.notify(&t("pitch.videoFirst"), true);
            return;
        };
        let (x, y) = (f64::from(fx) * pitch::LENGTH, f64::from(fy) * pitch::WIDTH);
        let (x, y) = pitch::snap(x, y, 4.0).unwrap_or((x, y));
        state.2.push([u, v, x, y]);
    }

    /// The last pair taken back.
    pub fn pitch_calibrate_undo(&mut self) {
        if let Some(state) = self.calibrating.as_mut() {
            if state.3.take().is_none() {
                state.2.pop();
            }
        }
    }

    /// The calibration kept: written beside the tracks, replacing one
    /// within a second of it.
    pub fn pitch_calibrate_done(&mut self) {
        use concat_vision::football::pitch;
        let Some((id, ms, pairs, _)) = self.calibrating.take() else {
            return;
        };
        if self.analysis_tool == 20 {
            self.analysis_tool = 0;
        }
        let from: Vec<(f64, f64)> = pairs.iter().map(|p| (p[0], p[1])).collect();
        let to: Vec<(f64, f64)> = pairs.iter().map(|p| (p[2], p[3])).collect();
        if pitch::homography(&from, &to).is_none() {
            self.notify(&t("pitch.badPoints"), true);
            return;
        }
        let Some(request) = self.football_request(&id) else {
            return;
        };
        let dir = request.dir();
        let mut calibration = pitch::Calibration::load(&dir);
        calibration
            .anchors
            .retain(|anchor| anchor.ms.abs_diff(ms) > 1000);
        calibration.anchors.push(pitch::Anchor { ms, pairs });
        match calibration.save(&dir) {
            Ok(()) => self.notify(&t("pitch.calibrated"), false),
            Err(error) => self.notify(&error, true),
        }
    }

    /// A dot clicked on the board: that player is the chosen one, as a
    /// click on him on the monitor makes him.
    pub fn pitch_dot_clicked(&mut self, id: i32) {
        if let Ok(id) = u32::try_from(id) {
            self.football_chosen = Some(id);
            self.request_preview();
        }
    }

    /// The chosen player's real number typed on the board: digits only,
    /// none to go back to the board's own count.
    pub fn pitch_player_number(&mut self, id: i32, text: &str) {
        let digits: String = text.chars().filter(char::is_ascii_digit).take(3).collect();
        let number = digits.parse::<u32>().ok();
        let squad = concat_vision::football::Squad::load(&self.eval_dir());
        let known = number.and_then(|n| squad.find(n).cloned());
        self.edit_roster(id, |player| {
            player.number = number;
            if let Some(known) = known {
                if player.name.trim().is_empty() {
                    player.name = known.name.clone();
                }
                if player.role.is_none() {
                    player.role = known.role;
                }
            }
        });
    }

    /// The chosen player's role picked on her card: 0 for none.
    pub fn pitch_player_role(&mut self, id: i32, index: i32) {
        let role = usize::try_from(index - 1)
            .ok()
            .and_then(|index| concat_vision::football::Role::ALL.get(index).copied());
        self.edit_roster(id, |player| player.role = role);
    }

    /// A dot dragged to where the player really is: her feet in the picture
    /// and that point of the pitch are a calibration pair. Into the
    /// calibration of this instant if there is one; else four points read
    /// off the map in force and this one, weighed twice; else the start of
    /// a calibration made of players, which counts from the fourth.
    pub fn pitch_dot_moved(&mut self, id: i32, fx: f32, fy: f32) {
        use concat_vision::football::pitch;
        let Ok(id) = u32::try_from(id) else {
            return;
        };
        let Some(clip) = self.board_clip() else {
            return;
        };
        let Some(dir) = self.football_request(&clip.id).map(|request| request.dir()) else {
            return;
        };
        let Some(set) = concat_vision::football::shared(&dir) else {
            return;
        };
        let seconds = self.source_at_playhead(&clip);
        let Some(tracked) = set.track_at(id, seconds) else {
            self.notify(&t("pitch.notHere"), true);
            return;
        };
        let (u, v) = pitch::feet(&tracked.bbox);
        let pair = [
            u,
            v,
            f64::from(fx.clamp(0.0, 1.0)) * pitch::LENGTH,
            f64::from(fy.clamp(0.0, 1.0)) * pitch::WIDTH,
        ];
        let ms = (seconds.max(0.0) * 1000.0).round() as u64;
        let mut calibration = pitch::Calibration::load(&dir);
        let motion = pitch::Motion::of(&set);
        let here = calibration
            .anchors
            .iter()
            .position(|anchor| anchor.ms.abs_diff(ms) <= 1000);
        let count = match here {
            Some(index) => {
                calibration.anchors[index].pairs.push(pair);
                calibration.anchors[index].pairs.len()
            }
            None => match pitch::map_at(&calibration, &motion, ms) {
                Some(map) => {
                    let mut pairs: Vec<[f64; 4]> =
                        [(0.25, 0.4), (0.75, 0.4), (0.8, 0.85), (0.2, 0.85)]
                            .into_iter()
                            .filter_map(|(a, b)| {
                                pitch::apply(&map, (a, b)).map(|(x, y)| [a, b, x, y])
                            })
                            .collect();
                    pairs.push(pair);
                    pairs.push(pair);
                    let count = pairs.len();
                    calibration.anchors.push(pitch::Anchor { ms, pairs });
                    count
                }
                None => {
                    calibration.anchors.push(pitch::Anchor {
                        ms,
                        pairs: vec![pair],
                    });
                    1
                }
            },
        };
        if let Err(error) = calibration.save(&dir) {
            self.notify(&error, true);
            return;
        }
        if count < 4 {
            self.notify(&tf("pitch.placedMore", &[&(4 - count)]), false);
        } else {
            self.notify(&t("pitch.placed"), false);
        }
        self.request_preview();
    }

    /// The chosen player's name typed on the board.
    pub fn pitch_player_name(&mut self, id: i32, text: &str) {
        let name: String = text.chars().take(40).collect();
        self.edit_roster(id, |player| player.name = name);
    }

    /// Player `id` of the board's footage renamed, and the board redrawn.
    fn edit_roster(&mut self, id: i32, change: impl FnOnce(&mut concat_vision::football::Player)) {
        let Ok(id) = u32::try_from(id) else {
            return;
        };
        let Some(request) = self
            .board_clip()
            .and_then(|clip| self.football_request(&clip.id))
        else {
            return;
        };
        let dir = request.dir();
        let mut roster = concat_vision::football::Roster::load(&dir);
        roster.edit(id, change);
        if let Err(error) = roster.save(&dir) {
            self.notify(&error, true);
        }
        // The judged team's player kept in the squad, so the next match
        // knows her by her number.
        let team = concat_vision::football::shared(&dir).and_then(|set| {
            set.instants_of(id)
                .into_iter()
                .map(|(_, tracked)| tracked.team)
                .find(|team| *team != 0)
        });
        if team == Some(self.eval_team)
            && let Some(player) = roster.players.get(&id)
            && let Some(number) = player.number
        {
            let library = self.eval_dir();
            let mut squad = concat_vision::football::Squad::load(&library);
            squad.upsert(number, &player.name, player.role);
            if let Err(error) = squad.save(&library) {
                log::warn!("football: could not keep the squad: {error}");
            }
        }
        self.request_preview();
    }

    /// The board, published.
    fn publish_board(&self, editor: &Editor<'_>, models: &Models) {
        let (dots, lines) = self.board_layers();
        sync(&models.pitch_dots, dots);
        sync(&models.pitch_lines, lines);
        sync(&models.pitch_pairs, self.board_pairs());
        // The legend and the badge from the cache the layers just filled.
        let state = self
            .board_clip()
            .and_then(|clip| self.board_state(&clip.id));
        sync(
            &models.pitch_kits,
            state
                .as_ref()
                .map(|(set, _, _, _)| Self::board_kits(set))
                .unwrap_or_default(),
        );
        let calibrated = state
            .as_ref()
            .is_some_and(|(_, _, calibration, _)| !calibration.anchors.is_empty());
        // The chosen player's card: his number and name, to be edited.
        let chosen = self.football_chosen.and_then(|id| {
            let (_, _, _, numbers) = state.as_ref()?;
            let (number, name, role) = numbers.get(&id).cloned().unwrap_or_default();
            editor.set_pitch_chosen_role(role);
            Some((id, number, name))
        });
        sync(
            &models.pitch_roles,
            std::iter::once(SharedString::from(t("pitch.noRole")))
                .chain(
                    concat_vision::football::Role::ALL
                        .iter()
                        .map(|role| SharedString::from(role.label())),
                )
                .collect(),
        );
        editor.set_pitch_chosen_id(chosen.as_ref().map_or(-1, |(id, _, _)| *id as i32));
        editor.set_pitch_chosen_number(
            chosen
                .as_ref()
                .filter(|(_, number, _)| *number > 0)
                .map(|(_, number, _)| number.to_string())
                .unwrap_or_default()
                .into(),
        );
        editor.set_pitch_chosen_name(chosen.map(|(_, _, name)| name).unwrap_or_default().into());
        editor.set_pitch_calibrated(calibrated);
        editor.set_pitch_calibrating(self.calibrating.is_some());
        editor.set_pitch_pairs(
            self.calibrating
                .as_ref()
                .map_or(0, |state| state.2.len() as i32),
        );
        editor.set_pitch_waiting(
            self.calibrating
                .as_ref()
                .is_some_and(|state| state.3.is_some()),
        );
    }

    // ── the analysis toolbar ──
    //
    // The monitor's own toolbar, the way Metrica Play and KlipDraw put the
    // tools on the picture rather than in a panel: one tool out at a time,
    // and the picture is where it is used. A player tool is a click on the
    // player - the analysis runs first if the footage has none, and the
    // click lands when it is done; a drawing tool is a drag, or a click per
    // point and Finish. One colour for all of them, cycled on the toolbar.
    // The inspector's sections remain for the knobs.

    /// The toolbar's player tools by their ids, as mark styles.
    fn mark_tool(&self) -> Option<model::MarkStyle> {
        match self.analysis_tool {
            1 => Some(model::MarkStyle::Spotlight),
            2 => Some(model::MarkStyle::Shade),
            3 => Some(model::MarkStyle::Ring),
            4 => Some(model::MarkStyle::Arrow),
            5 => Some(model::MarkStyle::Frame),
            6 => Some(model::MarkStyle::Erase),
            _ => None,
        }
    }

    /// A tool picked on the toolbar: 0 the pointer; 1 to 6 the player
    /// tools (spotlight, shade, ring, arrow, frame, erase); 10 to 14 the
    /// drawings (line, arrow, curve, zone, link). Picking the tool already
    /// out puts it away.
    pub fn analysis_tool_pick(&mut self, id: i32) {
        let id = if id == self.analysis_tool { 0 } else { id };
        self.calibrating = None;
        self.analysis_tool = id;
        self.draw_draft = None;
        self.painting = false;
        match id {
            10..=14 => {
                self.draw_tool = Some((id - 10) as usize);
                self.football_picking = false;
            }
            1..=6 => {
                self.draw_tool = None;
                // The players outlined, so the eye finds what to click.
                self.football_picking = true;
            }
            _ => {
                self.draw_tool = None;
                self.football_picking = false;
            }
        }
        self.request_preview();
    }

    /// The toolbar's colour, the next of the palette round; the chosen
    /// player's marks on the selected clip take it too.
    pub fn analysis_colour_next(&mut self) {
        self.analysis_colour = (self.analysis_colour + 1) % Self::MARK_COLOURS.len();
        self.draw_style.colour = Self::MARK_COLOURS[self.analysis_colour];
        if self.football_chosen.is_some() {
            self.football_colour(self.analysis_colour as i32);
        }
    }

    /// Every mark and drawing off the selected clip, as one undo step.
    pub fn analysis_clear(&mut self) {
        self.draw_draft = None;
        let Some(id) = self.sole_selection() else {
            return;
        };
        let Some(clip) = self.clip(&id).cloned() else {
            return;
        };
        let mut commands = Vec::new();
        if !clip.football.is_empty() {
            commands.push(Command::SetClipFootball {
                clip_id: id.clone(),
                marks: Vec::new(),
            });
        }
        if !clip.drawings.is_empty() {
            commands.push(Command::SetClipDrawings {
                clip_id: id,
                drawings: Vec::new(),
            });
        }
        if !commands.is_empty() {
            self.apply(Command::Batch { commands });
        }
    }

    /// The picture a toolbar click acts on: the one selected if it is on
    /// the stage, else the topmost picture under the click, which becomes
    /// the selection.
    fn analysis_target(&mut self, x: f32, y: f32) -> Option<Clip> {
        if let Some(clip) = self.sole_selection().and_then(|id| self.clip(&id).cloned())
            && clip.kind.is_visual()
            && self
                .stage_clips()
                .into_iter()
                .any(|shown| shown.id == clip.id)
        {
            return Some(clip);
        }
        let id = self.stage_hit(f64::from(x), f64::from(y))?;
        let clip = self.clip(&id).cloned()?;
        if !clip.kind.is_visual() {
            return None;
        }
        self.selection = vec![id];
        Some(clip)
    }

    /// A press on the stage with a player tool out: the mark on the
    /// player under it, or off him when he wears it already. With no
    /// analysis yet the press is kept, the analysis started, and the mark
    /// put on when the players are found. `false` when no player tool is
    /// out, so the press does what it would have.
    fn mark_press(&mut self, x: f32, y: f32) -> bool {
        let Some(style) = self.mark_tool() else {
            return false;
        };
        let Some(clip) = self.analysis_target(x, y) else {
            return true;
        };
        let Some(request) = self.football_request(&clip.id) else {
            self.notify(&t("football.needsFootage"), true);
            return true;
        };
        let [u, v] = self.stage_to_source(&clip, f64::from(x), f64::from(y));
        let at = self.source_at_playhead(&clip);
        if !request.analysed() {
            self.football_pending = Some((clip.id.clone(), u as f32, v as f32, at, style));
            if self.football_job.is_none() {
                self.football_analyse();
            }
            return true;
        }
        self.mark_player_at(&clip.id, u as f32, v as f32, at, style);
        true
    }

    /// The mark `style` on, or off, the player at source point `(u, v)` at
    /// source instant `at` of clip `id`.
    fn mark_player_at(&mut self, id: &str, u: f32, v: f32, at: f64, style: model::MarkStyle) {
        let Some(request) = self.football_request(id) else {
            return;
        };
        let Some(set) = concat_vision::football::shared(&request.dir()) else {
            return;
        };
        let Some(tracked) = set.pick(at, u, v) else {
            self.notify(&t("football.nobodyThere"), true);
            return;
        };
        self.football_chosen = Some(tracked.id);
        let Some(clip) = self.clip(id).cloned() else {
            return;
        };
        let mut marks = clip.football.clone();
        match marks
            .iter()
            .position(|mark| mark.track == tracked.id && mark.style == style)
        {
            Some(index) => {
                marks.remove(index);
            }
            None => {
                let mut mark = model::FootballMark::new(tracked.id, style);
                mark.colour = Self::MARK_COLOURS[self.analysis_colour];
                marks.push(mark);
            }
        }
        self.apply(Command::SetClipFootball {
            clip_id: id.to_owned(),
            marks,
        });
    }

    /// The press kept while the players were being found, put on now.
    fn land_pending_mark(&mut self) {
        if let Some((id, u, v, at, style)) = self.football_pending.take() {
            self.mark_player_at(&id, u, v, at, style);
        }
    }

    /// The analysis workspace, or back to the one it replaced: the monitor
    /// large, the tagging pane and the inspector beside it, the timeline
    /// along the bottom - the layout every analysis tool opens with.
    pub fn toggle_analysis_workspace(&mut self) {
        match self.dock_before_analysis.take() {
            Some(before) => self.dock = before,
            None => {
                self.dock_before_analysis = Some(std::mem::replace(
                    &mut self.dock,
                    crate::dock::analysis_dock(),
                ));
            }
        }
    }

    /// Whether the analysis workspace is the one showing.
    pub fn analysis_workspace(&self) -> bool {
        self.dock_before_analysis.is_some()
    }

    /// The toolbar's state, published.
    fn publish_analysis(&self, editor: &Editor<'_>) {
        let selected = self.sole_selection().and_then(|id| self.clip(&id).cloned());
        let available = selected.as_ref().is_some_and(|clip| clip.kind.is_visual())
            || self.stage_clips().iter().any(|clip| clip.kind.is_visual());
        editor.set_analysis_available(available);
        editor.set_analysis_tool(self.analysis_tool);
        let [r, g, b] = Self::MARK_COLOURS[self.analysis_colour];
        editor.set_analysis_colour(slint::Color::from_rgb_u8(r, g, b));
        editor.set_analysis_ready(
            selected
                .as_ref()
                .is_some_and(|clip| self.football_analysed(&clip.id)),
        );
        editor.set_analysis_progress(
            self.football_job
                .as_ref()
                .map_or(-1.0, |(_, _, fraction)| *fraction),
        );
        editor.set_analysis_silhouettes(
            self.football_job
                .as_ref()
                .is_some_and(|(what, _, _)| what == "silhouettes"),
        );
        editor.set_analysis_points(
            self.draw_draft
                .as_ref()
                .map_or(0, |(_, anchors, _)| anchors.len() as i32),
        );
        editor.set_analysis_workspace(self.analysis_workspace());
    }

    // ── tagging ──
    //
    // Nacsport's way of marking a match: a button per category, pressed as
    // the play happens, cuts a register round the playhead - the category's
    // lead before it and its lag after - and the register in hand then takes
    // descriptors and a rating. Registers are the timeline's plays, so the
    // ruler shows them and the Montage menu walks them; the tagging pane
    // lists them, filters them and makes a presentation of what it shows.

    /// The template in force.
    fn tag_template(&self) -> model::TagTemplate {
        self.project().tag_template()
    }

    /// The template changed, as one undo step.
    fn set_tag_template(&mut self, template: model::TagTemplate) {
        self.apply(Command::SetTagTemplate { template });
    }

    /// A category button, or its key: a register of that category round
    /// the playhead, which becomes the one in hand.
    pub fn tag_press(&mut self, index: i32) {
        let template = self.tag_template();
        let Some(category) = usize::try_from(index)
            .ok()
            .and_then(|index| template.categories.get(index))
        else {
            return;
        };
        let at = f64::from(self.playhead);
        let start = (at - category.lead).max(0.0);
        let end = (at + category.lag).max(start + 0.1);
        let mut plays = self.timeline().plays.clone();
        let number = plays
            .iter()
            .filter_map(|play| play.id.strip_prefix('p')?.parse::<u32>().ok())
            .max()
            .unwrap_or(0)
            + 1;
        let same = plays
            .iter()
            .filter(|play| play.tag == category.name)
            .count()
            + 1;
        let id = format!("p{number}");
        plays.push(model::Play {
            id: id.clone(),
            name: format!("{} {same}", category.name),
            tag: category.name.clone(),
            start,
            end,
            descriptors: Vec::new(),
            rating: 0,
        });
        self.apply(Command::SetPlays { plays });
        self.tag_current = Some(id);
    }

    /// The register in hand: the last one made or picked, if it is still
    /// there.
    fn tag_in_hand(&self) -> Option<model::Play> {
        let id = self.tag_current.as_ref()?;
        self.timeline()
            .plays
            .iter()
            .find(|play| play.id == *id)
            .cloned()
    }

    /// Changes the register in hand with `edit`, as one undo step.
    fn edit_in_hand(&mut self, edit: impl FnOnce(&mut model::Play)) {
        let Some(id) = self.tag_current.clone() else {
            self.notify(&t("tag.tagFirst"), true);
            return;
        };
        let mut plays = self.timeline().plays.clone();
        let Some(play) = plays.iter_mut().find(|play| play.id == id) else {
            return;
        };
        edit(play);
        self.apply(Command::SetPlays { plays });
    }

    /// A descriptor on the register in hand, on or off.
    pub fn tag_descriptor(&mut self, index: i32) {
        let template = self.tag_template();
        let Some(name) = usize::try_from(index)
            .ok()
            .and_then(|index| template.descriptors.get(index))
            .cloned()
        else {
            return;
        };
        self.edit_in_hand(|play| {
            if let Some(at) = play.descriptors.iter().position(|held| *held == name) {
                play.descriptors.remove(at);
            } else {
                play.descriptors.push(name);
            }
        });
    }

    /// The register in hand rated `stars`, or unrated when it already is.
    pub fn tag_rate(&mut self, stars: i32) {
        let stars = stars.clamp(0, 5) as u8;
        self.edit_in_hand(|play| {
            play.rating = if play.rating == stars { 0 } else { stars };
        });
    }

    /// A register picked from the list: in hand, the playhead at its start
    /// and its stretch marked.
    pub fn tag_select(&mut self, id: &str) {
        let Some(play) = self
            .timeline()
            .plays
            .iter()
            .find(|play| play.id == id)
            .cloned()
        else {
            return;
        };
        self.tag_current = Some(play.id);
        self.seek(play.start as f32);
        self.mark_in = Some(play.start as f32);
        self.mark_out = Some(play.end as f32);
    }

    /// A register deleted.
    pub fn tag_delete(&mut self, id: &str) {
        let mut plays = self.timeline().plays.clone();
        plays.retain(|play| play.id != id);
        if self.tag_current.as_deref() == Some(id) {
            self.tag_current = None;
        }
        self.apply(Command::SetPlays { plays });
    }

    /// The list shows one category, by its row, or all of them at -1.
    pub fn tag_filter_category(&mut self, index: i32) {
        let template = self.tag_template();
        self.tag_filter = usize::try_from(index)
            .ok()
            .and_then(|index| template.categories.get(index))
            .map(|category| category.name.clone());
    }

    /// The list shows registers rated at least `stars`.
    pub fn tag_filter_rating(&mut self, stars: i32) {
        self.tag_min_rating = stars.clamp(0, 5) as u8;
    }

    /// The registers the list shows, by when they happen.
    fn tag_filtered(&self) -> Vec<model::Play> {
        let mut plays: Vec<model::Play> = self
            .timeline()
            .plays
            .iter()
            .filter(|play| self.tag_filter.as_ref().is_none_or(|tag| play.tag == *tag))
            .filter(|play| play.rating >= self.tag_min_rating)
            .cloned()
            .collect();
        plays.sort_by(|a, b| a.start.total_cmp(&b.start));
        plays
    }

    /// The registers the list shows, made into a presentation: a timeline
    /// of their own, end to end.
    pub fn tag_present(&mut self) {
        let plays = self.tag_filtered();
        let name = match &self.tag_filter {
            Some(tag) => tf("montage.highlightsOf", &[tag]),
            None => t("tag.presentation"),
        };
        self.build_highlights_from(plays, name);
    }

    /// The template editor open or shut.
    pub fn tag_edit(&mut self, on: bool) {
        self.tag_editing = on;
    }

    /// A category added to the template, in the next colour of the round.
    pub fn tag_category_add(&mut self, name: &str) {
        const ROUND: [[u8; 3]; 8] = [
            [0x34, 0x98, 0xdb],
            [0x2e, 0xcc, 0x71],
            [0x9b, 0x59, 0xb6],
            [0xe6, 0x7e, 0x22],
            [0x1a, 0xbc, 0x9c],
            [0xe7, 0x4c, 0x3c],
            [0xf1, 0xc4, 0x0f],
            [0x95, 0xa5, 0xa6],
        ];
        let mut template = self.tag_template();
        let colour = ROUND[template.categories.len() % ROUND.len()];
        template
            .categories
            .push(model::TagCategory::new(name, colour));
        self.set_tag_template(template);
    }

    /// A category taken off the template; its registers stay.
    pub fn tag_category_remove(&mut self, index: i32) {
        let mut template = self.tag_template();
        if let Ok(index) = usize::try_from(index)
            && index < template.categories.len()
        {
            template.categories.remove(index);
            self.set_tag_template(template);
        }
    }

    /// A category's reach: `which` 0 before the press, 1 after it.
    pub fn tag_category_time(&mut self, index: i32, which: i32, seconds: f32) {
        let mut template = self.tag_template();
        let Some(category) = usize::try_from(index)
            .ok()
            .and_then(|index| template.categories.get_mut(index))
        else {
            return;
        };
        let seconds = f64::from(seconds).clamp(0.0, 60.0);
        if which == 0 {
            category.lead = seconds;
        } else {
            category.lag = seconds;
        }
        self.apply_in(Some("tag-time"), Command::SetTagTemplate { template });
    }

    /// A descriptor added to the template.
    pub fn tag_descriptor_add(&mut self, name: &str) {
        let mut template = self.tag_template();
        template.descriptors.push(name.to_owned());
        self.set_tag_template(template);
    }

    /// A descriptor taken off the template; registers keep it.
    pub fn tag_descriptor_remove(&mut self, index: i32) {
        let mut template = self.tag_template();
        if let Ok(index) = usize::try_from(index)
            && index < template.descriptors.len()
        {
            template.descriptors.remove(index);
            self.set_tag_template(template);
        }
    }

    /// The folder an exchange file is offered in: the project's, else home.
    fn exchange_dir(&self) -> std::path::PathBuf {
        self.session
            .as_ref()
            .map(|session| std::path::PathBuf::from(session.path()))
            .or_else(std::env::home_dir)
            .unwrap_or_default()
    }

    /// The registers the list shows, written where the person says as a
    /// Sportscode XML edit list (`xml`) - what Nacsport, Sportscode and
    /// LongoMatch import - or as CSV for a spreadsheet.
    pub fn tag_export(&mut self, xml: bool) {
        let plays = self.tag_filtered();
        if plays.is_empty() {
            self.notify(&t("montage.noPlays"), true);
            return;
        }
        let template = self.tag_template();
        let text = if xml {
            concat_project::exchange::to_sportscode(&plays, &template)
        } else {
            concat_project::exchange::to_csv(&plays)
        };
        let extension = if xml { "xml" } else { "csv" };
        let suggested = format!("{}.{extension}", self.timeline().name);
        let title = t(if xml {
            "tag.exportXml"
        } else {
            "tag.exportCsv"
        });
        let family = if xml { "Sportscode XML" } else { "CSV" }.to_owned();
        let directory = self.exchange_dir();
        let count = plays.len();
        crate::host::after_dialog(
            move || {
                let chosen = crate::platform::save_file(
                    &title,
                    &directory,
                    &suggested,
                    (family.as_str(), &[extension]),
                )?;
                Some(
                    if chosen
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case(extension))
                    {
                        chosen
                    } else {
                        chosen.with_extension(extension)
                    },
                )
            },
            move |studio, _, _, file: std::path::PathBuf| match std::fs::write(&file, text) {
                Ok(()) => studio.notify(
                    &tf("tag.exported", &[&count, &file.display().to_string()]),
                    false,
                ),
                Err(error) => studio.notify(&format!("{}: {error}", file.display()), true),
            },
        );
    }

    /// Plays from a Sportscode XML edit list - Nacsport's, Sportscode's,
    /// LongoMatch's - added to the active timeline at the times the file
    /// gives, and any category the template lacks added to it, as one undo
    /// step.
    pub fn tag_import(&mut self) {
        let title = t("tag.importXml");
        crate::host::after_dialog(
            move || {
                crate::platform::pick_files(&title, Some(("Sportscode XML", &["xml"])))
                    .and_then(|paths| paths.into_iter().next())
            },
            move |studio, _, _, file: std::path::PathBuf| studio.tag_import_from(&file),
        );
    }

    fn tag_import_from(&mut self, file: &std::path::Path) {
        let text = match std::fs::read_to_string(file) {
            Ok(text) => text,
            Err(error) => {
                self.notify(&format!("{}: {error}", file.display()), true);
                return;
            }
        };
        let mut plays = self.timeline().plays.clone();
        let next = plays
            .iter()
            .filter_map(|play| play.id.strip_prefix('p')?.parse::<u32>().ok())
            .max()
            .unwrap_or(0)
            + 1;
        let imported = match concat_project::exchange::from_sportscode(&text, next) {
            Ok(imported) => imported,
            Err(error) => {
                self.notify(&error, true);
                return;
            }
        };
        let count = imported.plays.len();
        plays.extend(imported.plays);
        let mut template = self.tag_template();
        let before = template.categories.len();
        for category in imported.categories {
            if !template
                .categories
                .iter()
                .any(|held| held.name == category.name)
            {
                template.categories.push(category);
            }
        }
        let gesture = "tag-import";
        if template.categories.len() != before {
            self.apply_in(Some(gesture), Command::SetTagTemplate { template });
        }
        self.apply_in(Some(gesture), Command::SetPlays { plays });
        self.end_burst();
        self.notify(&tf("tag.imported", &[&count]), false);
    }

    /// The colour a play is drawn in: its category's, else its tag's from
    /// the Montage menu's round, else grey.
    fn play_colour(&self, template: &model::TagTemplate, tag: &str) -> [u8; 3] {
        template
            .categories
            .iter()
            .find(|category| category.name == tag)
            .map(|category| category.colour)
            .or_else(|| {
                Self::PLAY_TAGS
                    .iter()
                    .find(|(name, _)| *name == tag)
                    .map(|(_, colour)| *colour)
            })
            .unwrap_or([0x95, 0xa5, 0xa6])
    }

    /// The tagging pane's buttons.
    fn tag_buttons(&self) -> Vec<TagButtonData> {
        let template = self.tag_template();
        let plays = &self.timeline().plays;
        template
            .categories
            .iter()
            .enumerate()
            .map(|(index, category)| {
                let [r, g, b] = category.colour;
                TagButtonData {
                    name: category.name.clone().into(),
                    colour: slint::Color::from_rgb_u8(r, g, b),
                    key: if index < 9 {
                        (index + 1).to_string().into()
                    } else {
                        SharedString::new()
                    },
                    count: plays
                        .iter()
                        .filter(|play| play.tag == category.name)
                        .count() as i32,
                    lead: category.lead as f32,
                    lag: category.lag as f32,
                }
            })
            .collect()
    }

    /// The tagging pane's descriptors, lit where the register in hand has
    /// them.
    fn tag_chips(&self) -> Vec<TagChipData> {
        let held = self
            .tag_in_hand()
            .map(|play| play.descriptors)
            .unwrap_or_default();
        self.tag_template()
            .descriptors
            .into_iter()
            .map(|name| TagChipData {
                on: held.contains(&name),
                name: name.into(),
            })
            .collect()
    }

    /// The tagging pane's matrix: every register as a bar on its
    /// category's row, across the timeline, whatever the list's filter.
    fn tag_bars(&self) -> Vec<TagBarData> {
        let template = self.tag_template();
        self.timeline()
            .plays
            .iter()
            .filter_map(|play| {
                let row = template
                    .categories
                    .iter()
                    .position(|category| category.name == play.tag)?;
                let [r, g, b] = template.categories[row].colour;
                Some(TagBarData {
                    row: row as i32,
                    start: play.start as f32,
                    end: play.end as f32,
                    colour: slint::Color::from_rgb_u8(r, g, b),
                    id: play.id.clone().into(),
                    current: self.tag_current.as_deref() == Some(play.id.as_str()),
                })
            })
            .collect()
    }

    /// The tagging pane's list.
    fn tag_rows(&self) -> Vec<TagRowData> {
        let template = self.tag_template();
        self.tag_filtered()
            .into_iter()
            .map(|play| {
                let [r, g, b] = self.play_colour(&template, &play.tag);
                TagRowData {
                    current: self.tag_current.as_deref() == Some(play.id.as_str()),
                    id: play.id.into(),
                    time: Self::montage_clock(play.start as f32).into(),
                    name: play.name.into(),
                    colour: slint::Color::from_rgb_u8(r, g, b),
                    descriptors: play.descriptors.join(" · ").into(),
                    rating: i32::from(play.rating),
                }
            })
            .collect()
    }

    /// Publishes the tagging pane.
    fn publish_tagging(&self, editor: &Editor<'_>, models: &Models) {
        sync(&models.tag_buttons, self.tag_buttons());
        sync(&models.tag_chips, self.tag_chips());
        sync(&models.tag_rows, self.tag_rows());
        sync(&models.tag_bars, self.tag_bars());
        let template = self.tag_template();
        let options: Vec<SharedString> = std::iter::once(t("tag.all").into())
            .chain(
                template
                    .categories
                    .iter()
                    .map(|category| category.name.clone().into()),
            )
            .collect();
        sync(&models.tag_filter_options, options);
        editor.set_tag_filter(
            self.tag_filter
                .as_ref()
                .and_then(|tag| {
                    template
                        .categories
                        .iter()
                        .position(|category| category.name == *tag)
                })
                .map_or(-1, |index| index as i32),
        );
        editor.set_tag_min_rating(i32::from(self.tag_min_rating));
        editor.set_tag_editing(self.tag_editing);
        let in_hand = self.tag_in_hand();
        editor.set_tag_current(
            in_hand
                .as_ref()
                .map(|play| play.name.clone())
                .unwrap_or_default()
                .into(),
        );
        editor.set_tag_rating(in_hand.map_or(0, |play| i32::from(play.rating)));
        self.publish_evaluation(editor, models);
        self.publish_assistant(editor, models);
    }

    // ── the post-match report ──
    //
    // The staff's report, as their slides lay it out: a page per phase -
    // build-up, press, defensive transition, the block - each saying what
    // worked, what did not and why, in figures against the game model in
    // use, with a still of the best and the worst. Written as a page of its
    // own into the exports folder and opened.

    /// Where reports go: the work folder's exports, beside the library.
    fn report_dir(&self) -> std::path::PathBuf {
        self.library_root()
            .parent()
            .map(|work| work.join("05 Exportaciones"))
            .unwrap_or_else(|| std::path::PathBuf::from(r"D:\Concat Futbol\05 Exportaciones"))
    }

    /// Today as the report names it, from the clock alone.
    fn report_date() -> String {
        let days = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_secs() / 86_400) as i64;
        // Howard Hinnant's civil-from-days.
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = doy - (153 * mp + 2) / 5 + 1;
        let month = if mp < 10 { mp + 3 } else { mp - 9 };
        let year = yoe + era * 400 + i64::from(month <= 2);
        format!("{year:04}-{month:02}-{day:02}")
    }

    /// The report of this timeline's evaluated registers.
    pub fn report_build(&mut self) {
        use concat_vision::football::report;
        let library = self.eval_library();
        let Some(profile) = self.eval_profile_in_use(&library) else {
            return;
        };
        let plays = self.plays_in_order();
        let mut items = Vec::new();
        for play in &plays {
            let teaching = play
                .descriptors
                .iter()
                .any(|d| d == Self::LIBRARY_GOOD || d == Self::LIBRARY_BAD);
            if teaching {
                continue;
            }
            let Some(judged) = self.eval_play(play) else {
                continue;
            };
            let action_name = Self::eval_kind_of(&library, &play.tag)
                .map_or_else(|| play.tag.clone(), |(_, name, _)| name);
            items.push(report::Item {
                action_key: judged.key,
                action_name,
                play_id: play.id.clone(),
                play_name: play.name.clone(),
                start: play.start,
                verdict: judged.judgement.verdict,
                measures: judged.measures,
            });
        }
        if items.is_empty() {
            self.notify(&t("report.nothing"), true);
            return;
        }
        let phases = report::build(&items, &profile);
        let names: HashMap<String, String> = items
            .iter()
            .map(|item| {
                (
                    item.play_id.clone(),
                    format!("{} · {}", item.play_name, Self::clock(item.start)),
                )
            })
            .collect();
        // A still of each phase's best and worst, read on a worker.
        let mut stills: Vec<(String, std::path::PathBuf, f64)> = Vec::new();
        for phase in &phases {
            for id in phase.best.iter().take(1).chain(phase.worst.iter().take(1)) {
                let Some(play) = plays.iter().find(|play| play.id == *id) else {
                    continue;
                };
                let middle = (play.start + play.end) / 2.0;
                let Some(clip) = self.footage_at(middle) else {
                    continue;
                };
                let Some(media) = self.project().media.iter().find(|m| m.id == clip.media_id)
                else {
                    continue;
                };
                let seconds = clip.source_start + (middle - clip.start) * clip.speed;
                stills.push((id.clone(), std::path::PathBuf::from(&media.path), seconds));
            }
        }
        // The model's KPIs over the footage under the board.
        let cards = self
            .board_clip()
            .and_then(|clip| self.eval_context(&clip.id))
            .map(|context| report::kpi_cards(&report::match_kpis(&context, self.eval_team)))
            .unwrap_or_default();
        let title = format!("Informe · {}", self.timeline().name);
        let subtitle = format!("{} · {}", profile.name, Self::report_date());
        let folder = self.report_dir().join(format!(
            "Informe {} {}",
            Self::report_date(),
            self.timeline()
                .name
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == ' ')
                .collect::<String>()
        ));
        self.notify(&t("report.writing"), false);
        std::thread::spawn(move || {
            let written = (|| -> Result<std::path::PathBuf, String> {
                std::fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
                let mut thumbs = HashMap::new();
                for (id, path, seconds) in stills {
                    let file = format!("{id}.jpg");
                    if Self::report_still(&path, seconds, &folder.join(&file)).is_ok() {
                        thumbs.insert(id, file);
                    }
                }
                let html =
                    report::to_html_with(&title, &subtitle, &cards, &phases, &thumbs, &names);
                let page = folder.join("informe.html");
                std::fs::write(&page, html).map_err(|error| error.to_string())?;
                Ok(page)
            })();
            crate::host::on_ui(move |studio, _, _| match written {
                Ok(page) => {
                    let _ = opener::open(&page);
                    studio.notify(&tf("report.written", &[&page.display()]), false);
                }
                Err(error) => studio.notify(&error, true),
            });
        });
    }

    /// One frame of `path` at `seconds`, 640 wide, as a JPEG.
    fn report_still(
        path: &std::path::Path,
        seconds: f64,
        out: &std::path::Path,
    ) -> Result<(), String> {
        use concat_core::time::Rational;
        use concat_media::{DecodeOptions, Decoder, FrameSource};
        let summary = media::probe(&path.to_string_lossy())?;
        let (w, h) = summary.video.as_ref().map_or((1920, 1080), |video| {
            (video.width.max(1), video.height.max(1))
        });
        let height = ((640.0 * f64::from(h) / f64::from(w)).round() as u32).max(2) & !1;
        let options = DecodeOptions::default()
            .starting_at(Rational::new((seconds.max(0.0) * 1000.0) as i64, 1000))
            .scaled_to(640, height)
            .limited_to(1);
        let mut decoder = Decoder::open(path, &options).map_err(|error| error.to_string())?;
        let frame = decoder
            .next_frame()
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "no frame".to_string())?;
        let rgba =
            image::RgbaImage::from_raw(frame.width(), frame.height(), frame.pixels().to_vec())
                .ok_or_else(|| "unexpected frame".to_string())?;
        image::DynamicImage::ImageRgba8(rgba)
            .to_rgb8()
            .save_with_format(out, image::ImageFormat::Jpeg)
            .map_err(|error| error.to_string())
    }

    // ── the reference library ──
    //
    // The coach's examples kept as files: a folder per action under
    // `02 Referencias`, each with `Así sí` and `Así no`. Import lays every
    // clip end to end on a timeline of its own, each under a register of
    // its action carrying its side; Analyse all runs the player analysis on
    // each in turn; Learn keeps every register whose clip is analysed and
    // calibrated as a reference and retrains once. Calibrating each clip -
    // four points - stays the coach's: nothing else can say where the
    // pitch is.

    /// The descriptor a library register carries: which side it teaches.
    const LIBRARY_GOOD: &'static str = "Así sí";
    const LIBRARY_BAD: &'static str = "Así no";
    /// And once it has been learned, so a second Learn does not count it twice.
    const LIBRARY_LEARNED: &'static str = "Aprendida";

    /// Where the library's folders are: beside the model's folder when that
    /// is the one on D:, else the usual place.
    fn library_root(&self) -> std::path::PathBuf {
        let strip = |path: std::path::PathBuf| {
            let text = path.to_string_lossy().into_owned();
            std::path::PathBuf::from(text.strip_prefix(r"\\?\").unwrap_or(&text).to_string())
        };
        std::fs::canonicalize(self.eval_dir())
            .ok()
            .map(strip)
            .and_then(|model| model.parent().map(|work| work.join("02 Referencias")))
            .filter(|root| root.is_dir())
            .unwrap_or_else(|| std::path::PathBuf::from(r"D:\Concat Futbol\02 Referencias"))
    }

    /// The action a library folder names: one of the four, matched loosely
    /// ("Bloque defensivo - basculación" is "Bloque defensivo / basculación").
    fn library_action_of(folder: &str) -> Option<String> {
        let plain = |s: &str| {
            s.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        };
        let folder = plain(folder);
        Self::EVAL_ACTIONS
            .iter()
            .map(|action| action.name().to_string())
            .find(|name| plain(name) == folder)
    }

    /// The team a library clip's file names: «equipo 1» or «equipo 2», or a
    /// shirt colour matched to the nearer of the two kits the analysis found.
    fn library_team_named(&self, clip: &Clip) -> Option<u8> {
        let media = self
            .project()
            .media
            .iter()
            .find(|m| m.id == clip.media_id)?;
        let name = std::path::Path::new(&media.path)
            .file_stem()?
            .to_string_lossy()
            .to_lowercase();
        if name.contains("equipo 1") || name.contains("equipo1") {
            return Some(1);
        }
        if name.contains("equipo 2") || name.contains("equipo2") {
            return Some(2);
        }
        let colours: [(&str, [f32; 3]); 12] = [
            ("rojo", [200.0, 40.0, 40.0]),
            ("roja", [200.0, 40.0, 40.0]),
            ("granate", [120.0, 30.0, 45.0]),
            ("azul", [40.0, 70.0, 170.0]),
            ("celeste", [120.0, 180.0, 230.0]),
            ("blanco", [235.0, 235.0, 235.0]),
            ("blanca", [235.0, 235.0, 235.0]),
            ("negro", [30.0, 30.0, 35.0]),
            ("amarillo", [230.0, 210.0, 50.0]),
            ("verde", [40.0, 150.0, 70.0]),
            ("naranja", [240.0, 130.0, 30.0]),
            ("morado", [110.0, 50.0, 140.0]),
        ];
        let wanted = colours
            .iter()
            .find(|(word, _)| {
                name.split(|c: char| !c.is_alphanumeric())
                    .any(|w| w == *word)
            })
            .map(|(_, rgb)| *rgb)?;
        let dir = self.football_request(&clip.id)?.dir();
        let set = concat_vision::football::shared(&dir)?;
        if set.team_colours.len() < 2 {
            return None;
        }
        let distance = |kit: &[u8; 3]| {
            (0..3)
                .map(|i| (f32::from(kit[i]) - wanted[i]).powi(2))
                .sum::<f32>()
        };
        Some(
            if distance(&set.team_colours[0]) <= distance(&set.team_colours[1]) {
                1
            } else {
                2
            },
        )
    }

    /// Import: every video in the library onto a new timeline, end to end,
    /// each under a register of its action marked with its side.
    pub fn library_import(&mut self) {
        use concat_vision::football::learn;
        if self.session.is_none() {
            return;
        }
        let root = self.library_root();
        let videos = ["mp4", "mov", "mkv", "avi", "m4v", "mts", "webm"];
        let listed = |dir: &std::path::Path| -> Vec<std::path::PathBuf> {
            let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
                .map(|entries| entries.filter_map(|e| e.ok()).map(|e| e.path()).collect())
                .unwrap_or_default();
            files.sort();
            files
        };
        // (action name, good, file); and the coach's own actions met on the way.
        let mut found: Vec<(String, bool, std::path::PathBuf)> = Vec::new();
        let mut own: Vec<String> = Vec::new();
        let mut folders: Vec<(String, std::path::PathBuf)> = Vec::new();
        for dir in listed(&root).into_iter().filter(|p| p.is_dir()) {
            let name = dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if name.starts_with('_') {
                for inner in listed(&dir).into_iter().filter(|p| p.is_dir()) {
                    let inner_name = inner
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    own.push(inner_name.clone());
                    folders.push((inner_name, inner));
                }
            } else if let Some(action) = Self::library_action_of(&name) {
                folders.push((action, dir));
            }
        }
        for (action, dir) in folders {
            for (side, good) in [(Self::LIBRARY_GOOD, true), (Self::LIBRARY_BAD, false)] {
                for file in listed(&dir.join(side)) {
                    let video = file
                        .extension()
                        .map(|e| e.to_string_lossy().to_lowercase())
                        .is_some_and(|e| videos.contains(&e.as_str()));
                    if video {
                        found.push((action.clone(), good, file));
                    }
                }
            }
        }
        if found.is_empty() {
            self.notify(&tf("library.empty", &[&root.display()]), true);
            return;
        }
        // What the project already holds is not laid down again: a second
        // import brings only the clips added since.
        let held: HashSet<String> = self
            .project()
            .media
            .iter()
            .map(|m| m.path.to_lowercase())
            .collect();
        found.retain(|(_, _, file)| !held.contains(&file.to_string_lossy().to_lowercase()));
        if found.is_empty() {
            self.notify(&t("library.nothingNew"), false);
            return;
        }
        // The coach's own actions the library names, kept as actions whose
        // window is the whole clip.
        let library = self.eval_library();
        let mut actions = library.actions.clone();
        let before = actions.len();
        for name in own {
            if !actions.iter().any(|a| a.name.eq_ignore_ascii_case(&name)) {
                let key: String = name
                    .to_lowercase()
                    .chars()
                    .map(|c| if c.is_alphanumeric() { c } else { '_' })
                    .collect();
                actions.push(learn::ActionDef {
                    key,
                    name,
                    trigger: learn::Trigger::Manual,
                    window_s: 15.0,
                    kpis: learn::GENERIC_KPIS.iter().map(|k| k.to_string()).collect(),
                });
            }
        }
        if actions.len() != before
            && let Err(error) = learn::save_actions(&self.eval_dir(), &actions)
        {
            self.notify(&error, true);
        }
        let count = found.len();
        self.notify(&tf("library.probing", &[&count]), false);
        spawn_in_project(
            move || {
                found
                    .into_iter()
                    .map(|(action, good, file)| {
                        let summary = media::probe(&file.to_string_lossy());
                        (action, good, summary)
                    })
                    .collect::<Vec<_>>()
            },
            |studio, _, _, probed| studio.library_lay_out(probed),
        );
    }

    /// The probed library on a timeline of its own.
    fn library_lay_out(
        &mut self,
        probed: Vec<(
            String,
            bool,
            Result<concat_host::media::MediaSummary, String>,
        )>,
    ) {
        let gesture = "library-import";
        self.apply_in(Some(gesture), Command::AddTimeline);
        let timeline_id = self.timeline().id.clone();
        self.apply_in(
            Some(gesture),
            Command::RenameTimeline {
                timeline_id,
                name: t("library.timeline"),
            },
        );
        let Some(lane) = self.timeline().tracks.first().map(|track| track.id.clone()) else {
            self.end_burst();
            return;
        };
        // The categories, as an evaluation run lays them.
        let library = self.eval_library();
        let mut template = self.tag_template();
        let names: Vec<String> = Self::EVAL_ACTIONS
            .iter()
            .map(|a| a.name().to_string())
            .chain(library.actions.iter().map(|a| a.name.clone()))
            .collect();
        let palette = [
            [0xe7, 0x4c, 0x3c],
            [0x34, 0x98, 0xdb],
            [0x2e, 0xcc, 0x71],
            [0x9b, 0x59, 0xb6],
            [0xe6, 0x7e, 0x22],
            [0x1a, 0xbc, 0x9c],
        ];
        let had = template.categories.len();
        for (index, name) in names.into_iter().enumerate() {
            if !template.categories.iter().any(|c| c.name == name) {
                template.categories.push(model::TagCategory {
                    name,
                    colour: palette[index % palette.len()],
                    lead: 0.0,
                    lag: 10.0,
                });
            }
        }
        if template.categories.len() != had {
            self.apply_in(Some(gesture), Command::SetTagTemplate { template });
        }
        let mut plays: Vec<model::Play> = Vec::new();
        let mut cursor = 0.0;
        let mut failed = 0;
        let mut number = self
            .project()
            .timelines
            .iter()
            .flat_map(|timeline| timeline.plays.iter())
            .filter_map(|play| play.id.strip_prefix('p')?.parse::<u32>().ok())
            .max()
            .unwrap_or(0);
        for (action, good, summary) in probed {
            let Ok(summary) = summary else {
                failed += 1;
                continue;
            };
            let item = summary.to_new_media();
            let path = item.path.clone();
            let media_id = self
                .apply_in(Some(gesture), Command::AddMedia { item })
                .or_else(|| {
                    self.project()
                        .media
                        .iter()
                        .find(|m| m.path == path)
                        .map(|m| m.id.clone())
                });
            let Some(media_id) = media_id else {
                failed += 1;
                continue;
            };
            let Some(clip_id) = self.apply_in(
                Some(gesture),
                Command::AddClip {
                    media_id,
                    track_id: lane.clone(),
                    start: cursor,
                    ripple: false,
                },
            ) else {
                failed += 1;
                continue;
            };
            let Some(duration) = self.clip(&clip_id).map(|clip| clip.duration) else {
                continue;
            };
            number += 1;
            let same = plays.iter().filter(|p| p.tag == action).count() + 1;
            let side = if good {
                Self::LIBRARY_GOOD
            } else {
                Self::LIBRARY_BAD
            };
            plays.push(model::Play {
                id: format!("p{number}"),
                name: format!("{action} {same} ({side})"),
                tag: action,
                start: cursor,
                end: cursor + duration,
                descriptors: vec![side.to_string()],
                rating: if good { 5 } else { 1 },
            });
            // A second between clips, so no register touches the next.
            cursor += duration + 1.0;
        }
        let laid = plays.len();
        self.apply_in(Some(gesture), Command::SetPlays { plays });
        self.end_burst();
        self.seek(0.0);
        if failed > 0 {
            self.notify(&tf("library.importedSome", &[&laid, &failed]), true);
        } else {
            self.notify(&tf("library.imported", &[&laid]), false);
        }
    }

    /// Analyse all: every video clip of this timeline not yet analysed, one
    /// after another.
    pub fn library_analyse_all(&mut self) {
        let pending: Vec<String> = self
            .timeline()
            .clips
            .iter()
            .filter(|clip| clip.kind == model::ClipKind::Video && !self.football_analysed(&clip.id))
            .map(|clip| clip.id.clone())
            .collect();
        if pending.is_empty() {
            self.notify(&t("library.allAnalysed"), false);
            return;
        }
        self.notify(&tf("library.analysing", &[&pending.len()]), false);
        self.library_queue = pending;
        self.library_next();
    }

    /// The next clip of the queue to the analysis, when it is free.
    pub fn library_next(&mut self) {
        if self.football_job.is_some() {
            return;
        }
        while let Some(id) = self.library_queue.first().cloned() {
            self.library_queue.remove(0);
            if self.clip(&id).is_some() && !self.football_analysed(&id) {
                self.selection = vec![id];
                self.football_analyse();
                return;
            }
        }
    }

    /// Learn: every library register whose clip is analysed and calibrated
    /// kept as a reference, and the models trained again, once.
    pub fn library_learn(&mut self) {
        use concat_vision::football::{eval::Reference, learn};
        let plays: Vec<model::Play> = self.timeline().plays.clone();
        let mut references = self.eval_library().references.clone();
        let (mut good, mut bad, mut waiting) = (0usize, 0usize, Vec::new());
        let mut learned_ids = Vec::new();
        for play in &plays {
            let side = if play.descriptors.iter().any(|d| d == Self::LIBRARY_GOOD) {
                true
            } else if play.descriptors.iter().any(|d| d == Self::LIBRARY_BAD) {
                false
            } else {
                continue;
            };
            if play.descriptors.iter().any(|d| d == Self::LIBRARY_LEARNED) {
                continue;
            }
            let Some(judged) = self.eval_play(play) else {
                waiting.push(play.name.clone());
                continue;
            };
            references.push(Reference {
                action_key: judged.key,
                good: side,
                measures: judged.measures,
            });
            learned_ids.push(play.id.clone());
            if side {
                good += 1;
            } else {
                bad += 1;
            }
        }
        if !learned_ids.is_empty() {
            let models = learn::retrain_all(&references);
            let written = self
                .eval_write("references.json", &references)
                .and_then(|()| self.eval_write("models.json", &models));
            if let Err(error) = written {
                self.notify(&error, true);
                return;
            }
            let mut plays = plays;
            for play in plays.iter_mut().filter(|p| learned_ids.contains(&p.id)) {
                play.descriptors.push(Self::LIBRARY_LEARNED.to_string());
            }
            self.apply(Command::SetPlays { plays });
        }
        let mut message = tf("library.learned", &[&good, &bad]);
        if !waiting.is_empty() {
            message.push_str(" · ");
            message.push_str(&tf(
                "library.waiting",
                &[&waiting.len(), &waiting.join(", ")],
            ));
        }
        self.notify(&message, good + bad == 0 && !waiting.is_empty());
    }

    // ── the assistant ──
    //
    // An assistant that works the analysis by being asked, in Spanish, by
    // text or by voice: a language model - a local one through Ollama, and
    // Claude when the coach has given a key - that answers with calls to the
    // tools below, each one a thing the window can already do. The model
    // runs on a worker thread; each call it makes is handed to this thread,
    // run here against the studio as it is, and its answer handed back.

    /// What the assistant is told it is and how to behave.
    const ASSISTANT_SYSTEM: &'static str = "Eres el asistente de análisis de Concat Fútbol, un editor de vídeo para analistas y entrenadores de fútbol. Respondes siempre en español, breve y concreto.

Actúa directamente: si la petición se entiende, llama a la herramienta sin pedir confirmación. Pregunta solo si falta un dato imprescindible. Nunca inventes datos ni resultados: los sacas de las herramientas.

Qué herramienta usar:
- «evalúa», «analiza el partido», «cómo lo hicimos» → `evaluar` (antes `configurar` si dicen perfil o equipo).
- Preguntas por un grupo de jugadas («qué presiones salieron mal», «cuáles salidas de balón fueron bien», «cuántas…») → `listar_jugadas` con `accion` y `veredicto`. Nunca uses `evaluar_jugada` para un grupo.
- Solo cuando nombran una jugada concreta por su id (p3, p12…): «por qué la p3…», «qué falló en la p3» → `evaluar_jugada` con ese id.
- «la pN está bien/mal hecha», «guárdala como ejemplo» → `marcar_referencia`.
- «sube/baja la exigencia», «el bloque no puede medir más de…» → `cambiar_umbral`; las claves de KPI válidas son: block_length (longitud o largo del bloque, lo que «mide» el bloque de delante a atrás, m), width (anchura del equipo de banda a banda, m), def_line_height (altura de la línea defensiva, m), line_gap (distancia entre líneas, m), shift_follow (seguimiento lateral, 0-1), pressers_3s (jugadores presionando a los 3 s), nearest_2s (distancia del más cercano a los 2 s, m), regain_s (segundos en recuperar), behind_ball_5s (jugadores por detrás del balón a los 5 s), reorganise_s (segundos en reorganizarse), support_15m (apoyos a 15 m), superiority_zone (superioridad cerca del balón), progress_m (metros progresados).
- En `cambiar_umbral` no toques `eliminatoria` salvo que lo pidan expresamente.
- «crea una acción…» → `crear_accion` con disparador perdida, recuperacion, posesion_rival, salida o manual.
- «móntame un vídeo/resumen…» → `crear_video`.
- «importa la biblioteca / los vídeos de referencia» → `importar_biblioteca`; «analiza todos los clips» → `analizar_todo`; «aprende de la biblioteca / de los ejemplos» → `aprender_biblioteca`.
- «hazme el informe / el post-partido» → `informe`.
- «qué sabes / qué has aprendido / reglas» → `ver_modelo`. Estado general → `estado`.

Acciones evaluables: Presión tras pérdida, Repliegue, Salida de balón, Bloque defensivo / basculación y las creadas por el cuerpo técnico. Veredictos: Bien, Mejorable, Mal, Sin datos. Para evaluar hacen falta jugadores analizados y el campograma calibrado: si una herramienta dice que falta, explica el paso al usuario.";

    /// The tools the assistant may call, with their arguments.
    fn assistant_tools() -> Vec<crate::assistant::Tool> {
        use serde_json::json;
        let tool =
            |name: &str, description: &str, schema: serde_json::Value| crate::assistant::Tool {
                name: name.to_string(),
                description: description.to_string(),
                schema,
            };
        let none = json!({"type": "object", "properties": {}});
        vec![
            tool(
                "estado",
                "Resumen del proyecto: vídeo en el cabezal, si está analizado y calibrado, modelo de juego y equipo evaluados, jugadas por acción y veredicto, referencias y modelos aprendidos.",
                none.clone(),
            ),
            tool(
                "analizar",
                "Analiza los jugadores y el balón del vídeo en el cabezal (tarda unos segundos; corre en segundo plano).",
                none.clone(),
            ),
            tool(
                "configurar",
                "Elige el modelo de juego (perfil) y/o el equipo evaluado (1 o 2, por el color de camiseta del campograma).",
                json!({"type": "object", "properties": {
                "perfil": {"type": "string", "description": "Nombre o parte del nombre del perfil"},
                "equipo": {"type": "integer", "enum": [1, 2]}}}),
            ),
            tool(
                "evaluar",
                "Encuentra y evalúa todas las acciones del equipo evaluado en el vídeo; crea una jugada por acción con su veredicto.",
                none.clone(),
            ),
            tool(
                "listar_jugadas",
                "Lista las jugadas de la línea de tiempo, filtrando por acción y/o veredicto.",
                json!({"type": "object", "properties": {
                "accion": {"type": "string"},
                "veredicto": {"type": "string", "enum": ["Bien", "Mejorable", "Mal", "Sin datos"]},
                "max": {"type": "integer"}}}),
            ),
            tool(
                "ir_a_jugada",
                "Selecciona una jugada y lleva el cabezal a su inicio.",
                json!({"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}),
            ),
            tool(
                "evaluar_jugada",
                "Veredicto y motivos (KPIs) de una jugada concreta.",
                json!({"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}),
            ),
            tool(
                "marcar_referencia",
                "Guarda una jugada como referencia de cómo sí (buena=true) o cómo no (buena=false) se hace su acción; la IA se reentrena.",
                json!({"type": "object", "properties": {
                "id": {"type": "string"}, "buena": {"type": "boolean"}}, "required": ["id", "buena"]}),
            ),
            tool(
                "ajustar_umbrales",
                "Ajusta los umbrales del modelo de juego de la acción de esa jugada a partir de sus referencias.",
                json!({"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}),
            ),
            tool(
                "cambiar_umbral",
                "Cambia una regla del modelo de juego en uso: mínimo y/o máximo de un KPI para una acción, y si es eliminatoria.",
                json!({"type": "object", "properties": {
                "accion": {"type": "string"},
                "kpi": {"type": "string", "description": "Clave del KPI. block_length = longitud del bloque, lo que mide de la línea más adelantada a la más retrasada (\"el bloque mide X m\"); width = anchura de banda a banda (\"ancho\"); def_line_height = altura de la línea defensiva; line_gap = distancia entre líneas; shift_follow = seguimiento lateral; pressers_3s = jugadores presionando; nearest_2s = distancia del más cercano; regain_s = segundos en recuperar; behind_ball_5s = jugadores por detrás del balón; reorganise_s = segundos en reorganizarse; support_15m = apoyos; superiority_zone = superioridad; progress_m = metros progresados."},
                "min": {"type": "number"}, "max": {"type": "number"},
                "eliminatoria": {"type": "boolean"}}, "required": ["accion", "kpi"]}),
            ),
            tool(
                "ver_modelo",
                "Reglas del modelo de juego en uso, referencias y lo aprendido por la IA, para una acción o todas.",
                json!({"type": "object", "properties": {"accion": {"type": "string"}}}),
            ),
            tool(
                "kpis",
                "Lista los KPIs que se pueden medir y usar en reglas o acciones nuevas.",
                none.clone(),
            ),
            tool(
                "crear_accion",
                "Crea una acción nueva del cuerpo técnico, que se detecta y evalúa como las demás.",
                json!({"type": "object", "properties": {
                "nombre": {"type": "string"},
                "disparador": {"type": "string", "enum": ["perdida", "recuperacion", "posesion_rival", "salida", "manual"]},
                "segundos": {"type": "number", "description": "Duración de la ventana"},
                "kpis": {"type": "array", "items": {"type": "string"}}}, "required": ["nombre", "disparador"]}),
            ),
            tool(
                "informe",
                "Genera el informe post-partido (por fases: qué funcionó, qué no funcionó, por qué pasó, con imágenes) de las jugadas evaluadas, con el modelo de juego en uso.",
                none.clone(),
            ),
            tool(
                "importar_biblioteca",
                "Importa los vídeos de la biblioteca de referencias (carpetas «Así sí» / «Así no» de cada acción en D:\\Concat Futbol\\02 Referencias) en una línea de tiempo nueva, ya etiquetados.",
                none.clone(),
            ),
            tool(
                "analizar_todo",
                "Analiza los jugadores de todos los clips de la línea de tiempo actual que falten, uno tras otro.",
                none.clone(),
            ),
            tool(
                "aprender_biblioteca",
                "Guarda como referencias todas las jugadas de la biblioteca ya analizadas y calibradas y reentrena la IA.",
                none.clone(),
            ),
            tool(
                "crear_video",
                "Monta un vídeo (línea de tiempo nueva) con las jugadas filtradas por acción y/o veredicto.",
                json!({"type": "object", "properties": {
                "accion": {"type": "string"},
                "veredicto": {"type": "string", "enum": ["Bien", "Mejorable", "Mal", "Sin datos"]},
                "nombre": {"type": "string"}}}),
            ),
        ]
    }

    /// The action an argument names: one of the four by name or key, or one
    /// of the coach's, matched loosely.
    fn assistant_action(library: &EvalLibrary, said: &str) -> Option<(String, String)> {
        let said = said.trim().to_lowercase().replace('_', " ");
        if said.is_empty() {
            return None;
        }
        Self::EVAL_ACTIONS
            .iter()
            .map(|action| (action.key().to_string(), action.name().to_string()))
            .chain(
                library
                    .actions
                    .iter()
                    .map(|def| (def.key.clone(), def.name.clone())),
            )
            .find(|(key, name)| {
                let name = name.to_lowercase();
                *key == said || name == said || name.contains(&said) || said.contains(&name)
            })
    }

    /// A timeline second as the coach reads it.
    fn clock(seconds: f64) -> String {
        let total = seconds.max(0.0).round() as u64;
        format!("{}:{:02}", total / 60, total % 60)
    }

    /// The verdict a register carries, if the evaluation gave it one.
    fn verdict_of(play: &model::Play) -> Option<&str> {
        ["Bien", "Mejorable", "Mal", "Sin datos"]
            .into_iter()
            .find(|label| play.descriptors.iter().any(|d| d == label))
    }

    /// One tool call, run against the studio; its answer is text for the
    /// model to read.
    pub fn assistant_tool(&mut self, name: &str, input: &serde_json::Value) -> String {
        use concat_vision::football::{eval::Profile, learn};
        let text = |key: &str| {
            input
                .get(key)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        let library = self.eval_library();
        match name {
            "estado" => {
                let clip = self.board_clip();
                let mut out = Vec::new();
                match &clip {
                    Some(clip) => {
                        out.push(format!("Vídeo: {}", clip.name));
                        let analysed = self.football_analysed(&clip.id);
                        out.push(format!(
                            "Jugadores analizados: {}",
                            if analysed { "sí" } else { "no" }
                        ));
                        let calibrated = analysed && self.eval_context(&clip.id).is_some();
                        out.push(format!(
                            "Campograma calibrado: {}",
                            if calibrated { "sí" } else { "no" }
                        ));
                    }
                    None => out.push("No hay vídeo en el cabezal ni seleccionado".to_string()),
                }
                if let Some(profile) = self.eval_profile_in_use(&library) {
                    out.push(format!("Modelo de juego: {}", profile.name));
                }
                out.push(format!(
                    "Perfiles disponibles: {}",
                    library
                        .profiles
                        .iter()
                        .map(|p| p.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
                out.push(format!("Equipo evaluado: {}", self.eval_team));
                let plays = &self.timeline().plays;
                let mut tally: std::collections::BTreeMap<(String, String), usize> =
                    Default::default();
                for play in plays.iter() {
                    let verdict = Self::verdict_of(play).unwrap_or("sin evaluar").to_string();
                    *tally.entry((play.tag.clone(), verdict)).or_default() += 1;
                }
                out.push(format!("Jugadas: {}", plays.len()));
                for ((tag, verdict), count) in tally {
                    out.push(format!("  {tag} · {verdict}: {count}"));
                }
                out.push(format!(
                    "Referencias guardadas: {}",
                    library.references.len()
                ));
                for model in &library.models {
                    out.push(format!(
                        "IA aprendida para {}: {} ejemplos, acierto {:.0} %",
                        model.action_key,
                        model.n_good + model.n_bad,
                        model.loo_accuracy.unwrap_or(0.0) * 100.0
                    ));
                }
                if !library.actions.is_empty() {
                    out.push(format!(
                        "Acciones propias: {}",
                        library
                            .actions
                            .iter()
                            .map(|a| a.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
                out.join("\n")
            }
            "analizar" => {
                let Some(clip) = self.board_clip() else {
                    return "No hay vídeo en el cabezal".into();
                };
                self.selection = vec![clip.id.clone()];
                self.football_analyse();
                "Análisis en marcha en segundo plano; tarda unos segundos por cada pocos segundos de vídeo.".into()
            }
            "configurar" => {
                let mut out = Vec::new();
                let wanted = text("perfil").to_lowercase().replace('_', " ");
                if !wanted.is_empty() {
                    match library
                        .profiles
                        .iter()
                        .position(|p| p.name.to_lowercase().contains(&wanted))
                    {
                        Some(index) => {
                            self.eval_profile = index;
                            out.push(format!("Modelo de juego: {}", library.profiles[index].name));
                        }
                        None => out.push(format!("No hay ningún perfil que se llame «{wanted}»")),
                    }
                }
                if let Some(team) = input
                    .get("equipo")
                    .and_then(|v| v.as_i64())
                    .and_then(|team| i32::try_from(team).ok())
                {
                    self.eval_team_pick(team);
                    out.push(format!("Equipo evaluado: {}", self.eval_team));
                }
                if out.is_empty() {
                    "Nada que cambiar".into()
                } else {
                    out.join("\n")
                }
            }
            "evaluar" => {
                self.eval_run();
                self.toast.message.to_string()
            }
            "listar_jugadas" => {
                let action =
                    Self::assistant_action(&library, &text("accion")).map(|(_, name)| name);
                let verdict = text("veredicto");
                let max = input.get("max").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
                let plays: Vec<String> = self
                    .plays_in_order()
                    .into_iter()
                    .filter(|play| action.as_ref().is_none_or(|a| play.tag == *a))
                    .filter(|play| {
                        verdict.is_empty() || Self::verdict_of(play) == Some(verdict.as_str())
                    })
                    .take(max)
                    .map(|play| {
                        format!(
                            "{} · {} · {}–{} · {}",
                            play.id,
                            play.name,
                            Self::clock(play.start),
                            Self::clock(play.end),
                            Self::verdict_of(&play).unwrap_or("sin evaluar")
                        )
                    })
                    .collect();
                if plays.is_empty() {
                    "No hay jugadas con ese filtro".into()
                } else {
                    plays.join("\n")
                }
            }
            "ir_a_jugada" => {
                let id = text("id");
                self.tag_select(&id);
                match self.tag_in_hand() {
                    Some(play) if play.id == id => {
                        format!("En {} ({})", play.name, Self::clock(play.start))
                    }
                    _ => format!("No existe la jugada {id}"),
                }
            }
            "evaluar_jugada" => {
                let id = text("id");
                self.tag_select(&id);
                let Some(play) = self.tag_in_hand().filter(|play| play.id == id) else {
                    return format!("No existe la jugada {id}");
                };
                match self.eval_play(&play) {
                    Some(judged) => {
                        let measured: Vec<String> = judged
                            .measures
                            .iter()
                            .filter_map(|(k, v)| v.map(|v| format!("{k}={v:.1}")))
                            .collect();
                        format!(
                            "{}: {}\nMotivos: {}\nKPIs: {}",
                            play.name,
                            judged.judgement.verdict.label(),
                            judged.judgement.reasons.join("; "),
                            measured.join(", ")
                        )
                    }
                    None => "Esa jugada no es evaluable: no es una de las acciones, o el vídeo no está analizado y calibrado".into(),
                }
            }
            "marcar_referencia" => {
                let id = text("id");
                self.tag_select(&id);
                if self.tag_in_hand().is_none_or(|play| play.id != id) {
                    return format!("No existe la jugada {id}");
                }
                let good = input.get("buena").and_then(|v| v.as_bool()).unwrap_or(true);
                self.eval_reference(good);
                self.toast.message.to_string()
            }
            "ajustar_umbrales" => {
                let id = text("id");
                self.tag_select(&id);
                self.eval_tune();
                self.toast.message.to_string()
            }
            "cambiar_umbral" => {
                let Some((key, action_name)) = Self::assistant_action(&library, &text("accion"))
                else {
                    return "No conozco esa acción".into();
                };
                // A key, or its Spanish name: the model sometimes says one.
                let said = text("kpi");
                let kpi = learn::GENERIC_KPIS
                    .iter()
                    .chain(
                        [
                            "pressers_3s",
                            "nearest_2s",
                            "regain_s",
                            "behind_ball_8s",
                            "reorganise_s",
                            "block_length_end",
                        ]
                        .iter(),
                    )
                    .find(|key| {
                        **key == said || learn::generic_label(key).eq_ignore_ascii_case(said.trim())
                    })
                    .map(|key| key.to_string());
                let Some(kpi) = kpi else {
                    return format!(
                        "No conozco el KPI «{said}». Claves válidas: {}",
                        learn::GENERIC_KPIS.join(", ")
                    );
                };
                let Some(mut profile) = self.eval_profile_in_use(&library) else {
                    return "No hay modelo de juego".into();
                };
                let rules = profile.rules.entry(key).or_default();
                let rule = match rules.iter().position(|r| r.kpi == kpi) {
                    Some(index) => &mut rules[index],
                    None => {
                        rules.push(concat_vision::football::eval::Rule {
                            kpi: kpi.clone(),
                            min: None,
                            max: None,
                            weight: 1.0,
                            major: false,
                        });
                        let last = rules.len() - 1;
                        &mut rules[last]
                    }
                };
                if let Some(min) = input.get("min").and_then(|v| v.as_f64()) {
                    rule.min = Some(min);
                }
                if let Some(max) = input.get("max").and_then(|v| v.as_f64()) {
                    rule.max = Some(max);
                }
                if let Some(major) = input.get("eliminatoria").and_then(|v| v.as_bool()) {
                    rule.major = major;
                }
                let summary = format!(
                    "{action_name} · {kpi}: mín. {:?}, máx. {:?}, eliminatoria {}",
                    rule.min, rule.max, rule.major
                );
                let mut profiles = library.profiles.clone();
                let index = self.eval_profile.min(profiles.len().saturating_sub(1));
                if let Some(slot) = profiles.get_mut(index) {
                    *slot = profile;
                }
                match Profile::save_all(&self.eval_dir(), &profiles) {
                    Ok(()) => summary,
                    Err(error) => error,
                }
            }
            "ver_modelo" => {
                let only = Self::assistant_action(&library, &text("accion")).map(|(key, _)| key);
                let Some(profile) = self.eval_profile_in_use(&library) else {
                    return "No hay modelo de juego".into();
                };
                let mut out = vec![format!("Modelo de juego: {}", profile.name)];
                for (key, rules) in &profile.rules {
                    if only.as_ref().is_some_and(|o| o != key) {
                        continue;
                    }
                    out.push(format!("{key}:"));
                    for rule in rules {
                        out.push(format!(
                            "  {} mín. {:?} máx. {:?} peso {} {}",
                            rule.kpi,
                            rule.min,
                            rule.max,
                            rule.weight,
                            if rule.major { "(eliminatoria)" } else { "" }
                        ));
                    }
                    let goods = library
                        .references
                        .iter()
                        .filter(|r| r.action_key == *key && r.good)
                        .count();
                    let bads = library
                        .references
                        .iter()
                        .filter(|r| r.action_key == *key && !r.good)
                        .count();
                    out.push(format!("  referencias: {goods} así sí, {bads} así no"));
                    if let Some(model) = library.models.iter().find(|m| m.action_key == *key) {
                        out.push(format!(
                            "  IA: acierto {:.0} % con {} ejemplos; pesos {:?}",
                            model.loo_accuracy.unwrap_or(0.0) * 100.0,
                            model.n_good + model.n_bad,
                            model
                                .kpis
                                .iter()
                                .zip(&model.weights)
                                .map(|(k, w)| format!("{k}:{w:+.2}"))
                                .collect::<Vec<_>>()
                        ));
                    }
                }
                out.join("\n")
            }
            "kpis" => learn::GENERIC_KPIS
                .iter()
                .map(|k| format!("{k}: {}", learn::generic_label(k)))
                .collect::<Vec<_>>()
                .join("\n"),
            "crear_accion" => {
                // «cierre_de_banda» from a model reads «Cierre de banda».
                let name = text("nombre").replace('_', " ");
                let mut letters = name.trim().chars();
                let name: String = match letters.next() {
                    Some(first) => first.to_uppercase().chain(letters).collect(),
                    None => return "Falta el nombre".into(),
                };
                let said = text("disparador").to_lowercase();
                let trigger = if said.contains("perd") {
                    learn::Trigger::BallLost
                } else if said.contains("recup") {
                    learn::Trigger::BallWon
                } else if said.contains("rival") {
                    learn::Trigger::RivalPossession { min_s: 4.0 }
                } else if said.contains("salida") {
                    learn::Trigger::OwnPossessionDeep {
                        max_depth: 35.0,
                        min_s: 2.0,
                    }
                } else {
                    learn::Trigger::Manual
                };
                let key: String = name
                    .to_lowercase()
                    .chars()
                    .map(|c| if c.is_alphanumeric() { c } else { '_' })
                    .collect();
                let kpis: Vec<String> = input
                    .get("kpis")
                    .and_then(|v| v.as_array())
                    .map(|list| {
                        list.iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_else(|| learn::GENERIC_KPIS.iter().map(|k| k.to_string()).collect());
                let mut actions = library.actions.clone();
                actions.retain(|a| a.key != key);
                actions.push(learn::ActionDef {
                    key: key.clone(),
                    name: name.clone(),
                    trigger,
                    window_s: input
                        .get("segundos")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(6.0),
                    kpis,
                });
                match learn::save_actions(&self.eval_dir(), &actions) {
                    Ok(()) => format!(
                        "Acción «{name}» creada (clave {key}). Se detectará al evaluar; márcale referencias para que la IA aprenda."
                    ),
                    Err(error) => error,
                }
            }
            "crear_video" => {
                let action =
                    Self::assistant_action(&library, &text("accion")).map(|(_, name)| name);
                let verdict = text("veredicto");
                let plays: Vec<model::Play> = self
                    .plays_in_order()
                    .into_iter()
                    .filter(|play| action.as_ref().is_none_or(|a| play.tag == *a))
                    .filter(|play| {
                        verdict.is_empty() || Self::verdict_of(play) == Some(verdict.as_str())
                    })
                    .collect();
                let count = plays.len();
                if count == 0 {
                    return "No hay jugadas con ese filtro".into();
                }
                let mut name = text("nombre");
                if name.is_empty() {
                    name = [action.clone().unwrap_or_default(), verdict.clone()]
                        .into_iter()
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>()
                        .join(" · ");
                    if name.is_empty() {
                        name = t("tag.presentation");
                    }
                }
                self.build_highlights_from(plays, name.clone());
                format!("Vídeo «{name}» montado con {count} jugadas en una línea de tiempo nueva")
            }
            "importar_biblioteca" => {
                self.library_import();
                format!("{} (se completa en segundo plano)", self.toast.message)
            }
            "analizar_todo" => {
                self.library_analyse_all();
                self.toast.message.to_string()
            }
            "informe" => {
                self.report_build();
                format!(
                    "{} El informe se abre solo en el navegador al terminar.",
                    self.toast.message
                )
            }
            "aprender_biblioteca" => {
                self.library_learn();
                self.toast.message.to_string()
            }
            other => format!("Herramienta desconocida: {other}"),
        }
    }

    /// The coach's message: shown at once, and the model asked on a worker.
    pub fn assistant_send(&mut self, text: &str) {
        let text = text.trim();
        if text.is_empty() || self.assistant_busy {
            return;
        }
        self.assistant_lines.push((0, text.to_string()));
        self.assistant_history.push(crate::assistant::Turn {
            role: crate::assistant::Role::User,
            parts: vec![crate::assistant::Part::Text(text.to_string())],
        });
        self.assistant_busy = true;
        let config = self.assistant_config();
        let mut history = self.assistant_history.clone();
        std::thread::spawn(move || {
            let tools = Self::assistant_tools();
            let mut exec = |name: &str, input: &serde_json::Value| -> String {
                let (tx, rx) = std::sync::mpsc::channel();
                let (name, input) = (name.to_string(), input.clone());
                crate::host::on_ui(move |studio, _, _| {
                    let _ = tx.send(studio.assistant_tool(&name, &input));
                });
                rx.recv_timeout(std::time::Duration::from_secs(120))
                    .unwrap_or_else(|_| "La app no respondió a tiempo".to_string())
            };
            let result = crate::assistant::run(
                &config,
                Self::ASSISTANT_SYSTEM,
                &mut history,
                &tools,
                &mut exec,
                8,
            );
            crate::host::on_ui(move |studio, _, _| {
                studio.assistant_busy = false;
                match result {
                    Ok(reply) => {
                        studio.assistant_history = history;
                        studio.assistant_provider_used = reply.provider.to_string();
                        let text = if reply.text.trim().is_empty() {
                            t("assistant.done")
                        } else {
                            reply.text
                        };
                        studio.assistant_lines.push((1, text));
                    }
                    Err(error) => {
                        // The question stays unanswered in the history: drop
                        // it, so the next one is not asked twice.
                        studio.assistant_history.pop();
                        studio.assistant_lines.push((2, error));
                    }
                }
            });
        });
    }

    /// The microphone: pressed once it listens; pressed again, what was
    /// said is written out by the transcriber the captions use - a
    /// multilingual model, so Spanish is heard - and sent as a message.
    pub fn assistant_voice(&mut self) {
        if let Some(recorder) = self.assistant_recorder.take() {
            let take = match recorder.stop() {
                Ok(take) => take,
                Err(error) => {
                    self.notify(&error, true);
                    return;
                }
            };
            let models = crate::panes::settings::installed(&self.settings.transcribers);
            let multilingual = |id: &str| !id.ends_with(".en");
            let chosen = self
                .prefs
                .transcriber_model
                .clone()
                .filter(|id| multilingual(id) && models.iter().any(|m| m.id == *id))
                .or_else(|| {
                    models
                        .iter()
                        .map(|m| m.id.clone())
                        .find(|id| multilingual(id))
                });
            let Some(model_id) = chosen else {
                self.notify(&t("assistant.voiceNeedsModel"), true);
                return;
            };
            let request = concat_speech::transcribe::TranscribeRequest {
                path: take.path.to_string_lossy().into_owned(),
                audio_stream: None,
                source_start: 0.0,
                window: take.seconds.max(0.2),
                model_id,
            };
            let dirs = self.host.dirs.clone();
            let transcriber = Arc::clone(&self.host.transcriber);
            self.assistant_hearing = true;
            std::thread::spawn(move || {
                let heard = transcriber.transcribe(&dirs, &request, |_| {});
                crate::host::on_ui(move |studio, _, _| {
                    studio.assistant_hearing = false;
                    match heard {
                        Ok(segments) => {
                            let text = segments
                                .iter()
                                .map(|segment| segment.text.trim())
                                .collect::<Vec<_>>()
                                .join(" ");
                            if text.trim().is_empty() {
                                studio.notify(&t("assistant.heardNothing"), true);
                            } else {
                                studio.assistant_send(&text);
                            }
                        }
                        Err(error) => studio.notify(&error, true),
                    }
                });
            });
        } else {
            if self.assistant_busy || self.assistant_hearing {
                return;
            }
            match concat_host::record::Recorder::start(self.eval_dir().join("voz.wav")) {
                Ok(recorder) => {
                    self.assistant_recorder = Some(recorder);
                    self.notify(&t("assistant.listening"), false);
                }
                Err(error) => self.notify(&error, true),
            }
        }
    }

    /// A fresh conversation.
    pub fn assistant_clear(&mut self) {
        if !self.assistant_busy {
            self.assistant_lines.clear();
            self.assistant_history.clear();
        }
    }

    /// The assistant's settings changed: which model, and the Claude key.
    pub fn assistant_settings(&mut self, provider: i32, model: &str, key: &str) {
        use crate::assistant::{Config, Provider};
        let dir = self.eval_dir();
        let mut config = Config::load(&dir);
        config.provider = match provider {
            1 => Provider::Local,
            2 => Provider::Claude,
            _ => Provider::Auto,
        };
        if !model.trim().is_empty() {
            config.ollama_model = model.trim().to_string();
        }
        if !key.trim().is_empty() {
            config.claude_key = Some(key.trim().to_string());
        }
        *self.assistant_config.borrow_mut() = None;
        match config.save(&dir) {
            Ok(()) => self.notify(&t("assistant.saved"), false),
            Err(error) => self.notify(&error, true),
        }
    }

    /// The assistant's settings, from the cache.
    fn assistant_config(&self) -> crate::assistant::Config {
        self.assistant_config
            .borrow_mut()
            .get_or_insert_with(|| crate::assistant::Config::load(&self.eval_dir()))
            .clone()
    }

    /// The assistant's pane.
    fn publish_assistant(&self, editor: &Editor<'_>, models: &Models) {
        use crate::assistant::Provider;
        let mut lines: Vec<AssistantLineData> = self
            .assistant_lines
            .iter()
            .map(|(role, text)| AssistantLineData {
                role: i32::from(*role),
                text: text.as_str().into(),
            })
            .collect();
        if lines.is_empty() {
            lines.push(AssistantLineData {
                role: 1,
                text: t("assistant.hello").into(),
            });
        }
        sync(&models.assistant_lines, lines);
        editor.set_assistant_busy(self.assistant_busy || self.assistant_hearing);
        editor.set_assistant_listening(self.assistant_recorder.is_some());
        let config = self.assistant_config();
        editor.set_assistant_provider(match config.provider {
            Provider::Auto => 0,
            Provider::Local => 1,
            Provider::Claude => 2,
        });
        editor.set_assistant_model(config.ollama_model.as_str().into());
        editor.set_assistant_has_key(config.has_claude());
        editor.set_assistant_used(self.assistant_provider_used.as_str().into());
    }

    // ── evaluation ──
    //
    // The analysis judged against the team's game model: the four collective
    // actions found in the footage, and any the coach defined, measured in
    // metres on the calibrated pitch and judged three ways - the chosen
    // profile's rules, the closeness to the reference examples, and the
    // model the app learned from those examples. The profiles, references,
    // learned models and own actions are the coach's library, kept in the
    // app's config rather than in a project, so what was taught on one
    // match judges the next.

    /// The four actions, in the order the template lists them.
    const EVAL_ACTIONS: [concat_vision::football::eval::Action; 4] = [
        concat_vision::football::eval::Action::PressAfterLoss,
        concat_vision::football::eval::Action::RecoveryRun,
        concat_vision::football::eval::Action::BuildUp,
        concat_vision::football::eval::Action::DefensiveBlock,
    ];

    /// The descriptor that marks a register the evaluation made, so a new
    /// run replaces its own and leaves the analyst's alone.
    const EVAL_AUTO: &'static str = "Auto";

    /// Where the library lives.
    pub fn eval_dir(&self) -> std::path::PathBuf {
        self.host.dirs.config.join("football")
    }

    /// The library, read again only when one of its files changes.
    pub fn eval_library(&self) -> Arc<EvalLibrary> {
        use concat_vision::football::{eval::Profile, learn};
        let dir = self.eval_dir();
        let stamp = |name: &str| {
            std::fs::metadata(dir.join(name))
                .and_then(|meta| meta.modified())
                .ok()
        };
        let key = [
            stamp("profiles.json"),
            stamp("references.json"),
            stamp("models.json"),
            stamp("actions.json"),
        ];
        let mut cache = self.eval_library_cache.borrow_mut();
        if let Some((held, library)) = cache.as_ref()
            && *held == key
        {
            return Arc::clone(library);
        }
        let mut profiles = Profile::load(&dir);
        if profiles.is_empty() {
            profiles = Profile::presets();
        }
        let library = Arc::new(EvalLibrary {
            profiles,
            references: std::fs::read(dir.join("references.json"))
                .ok()
                .and_then(|bytes| serde_json::from_slice(&bytes).ok())
                .unwrap_or_default(),
            models: learn::load_models(&dir),
            actions: learn::load_actions(&dir),
        });
        *cache = Some((key, Arc::clone(&library)));
        library
    }

    /// The profile in use.
    fn eval_profile_in_use(
        &self,
        library: &EvalLibrary,
    ) -> Option<concat_vision::football::eval::Profile> {
        library
            .profiles
            .get(self.eval_profile)
            .or_else(|| library.profiles.first())
            .cloned()
    }

    /// The analysis of clip `id` read onto the pitch, once per change of its
    /// tracks or its calibration.
    fn eval_context(&self, id: &str) -> Option<Arc<concat_vision::football::eval::Context>> {
        use concat_vision::football::{eval::Context, pitch::Calibration};
        let dir = self.football_request(id)?.dir();
        let set = concat_vision::football::shared(&dir)?;
        let stamp = |name: &str| {
            std::fs::metadata(dir.join(name))
                .and_then(|meta| meta.modified())
                .ok()
        };
        let key = (
            dir.clone(),
            stamp("tracks.json"),
            stamp("calibration.json"),
            stamp("roster.json"),
        );
        let mut cache = self.eval_context_cache.borrow_mut();
        if let Some((held, context)) = cache.as_ref()
            && *held == key
        {
            return Some(Arc::clone(context));
        }
        let calibration = Calibration::load(&dir);
        if calibration.anchors.is_empty() {
            return None;
        }
        let players = concat_vision::football::Roster::load(&dir).effective(&set);
        let context = Arc::new(Context::with_players(&set, &calibration, &players));
        *cache = Some((key, Arc::clone(&context)));
        Some(context)
    }

    /// The footage under timeline second `at`.
    fn footage_at(&self, at: f64) -> Option<Clip> {
        self.timeline()
            .clips
            .iter()
            .find(|clip| {
                clip.kind == model::ClipKind::Video
                    && clip.start <= at
                    && at < clip.start + clip.duration
            })
            .map(|clip| (**clip).clone())
    }

    /// The action a category names: its key, its name and, for one of the
    /// four, which.
    fn eval_kind_of(
        library: &EvalLibrary,
        tag: &str,
    ) -> Option<(
        String,
        String,
        Option<concat_vision::football::eval::Action>,
    )> {
        if let Some(action) = Self::EVAL_ACTIONS
            .into_iter()
            .find(|action| action.name() == tag)
        {
            return Some((
                action.key().to_string(),
                action.name().to_string(),
                Some(action),
            ));
        }
        library
            .actions
            .iter()
            .find(|def| def.name == tag)
            .map(|def| (def.key.clone(), def.name.clone(), None))
    }

    /// One window measured and judged three ways. The four actions are
    /// measured by their own KPIs with the generic ones beside them, a
    /// coach's action by the generic ones; the rules and references judge
    /// it, and the learned model, when there is one, may move the verdict a
    /// step.
    fn eval_window(
        &self,
        library: &EvalLibrary,
        context: &concat_vision::football::eval::Context,
        key: &str,
        action: Option<concat_vision::football::eval::Action>,
        start_ms: u64,
        end_ms: u64,
        team: u8,
    ) -> Option<Judged> {
        use concat_vision::football::{eval, learn};
        let profile = self.eval_profile_in_use(library)?;
        let window = learn::Window {
            key: key.to_string(),
            team,
            start_ms,
            end_ms,
        };
        let (generic, observability) = learn::measure_window(context, &window)?;
        let mut measures = match action {
            Some(action) => {
                let occurrence = eval::Occurrence {
                    action,
                    team,
                    start_ms,
                    end_ms,
                };
                context.assess(&occurrence, &profile, &[])?.0
            }
            None => eval::Measures::new(),
        };
        for (kpi, value) in generic {
            measures.entry(kpi).or_insert(value);
        }
        let references: Vec<eval::Reference> = library
            .references
            .iter()
            .filter(|r| r.action_key == key)
            .cloned()
            .collect();
        let mut judgement = eval::judge_key(key, &measures, &profile, &references, observability);
        if let Some(model) = library.models.iter().find(|model| model.action_key == key) {
            let prediction = model.predict(&measures);
            judgement.verdict = learn::blend(judgement.verdict, Some(&prediction));
            let accuracy = model
                .loo_accuracy
                .map(|a| format!(", acierto {:.0} %", a * 100.0))
                .unwrap_or_default();
            judgement.reasons.push(format!(
                "IA ({} ejemplos{accuracy}): {:.0} % parecido a vuestro «así sí»",
                model.n_good + model.n_bad,
                prediction.p_good * 100.0
            ));
            judgement.reasons.extend(model.explain(&prediction));
        }
        Some(Judged {
            key: key.to_string(),
            measures,
            judgement,
        })
    }

    /// A register measured and judged: its category names the action, the
    /// footage under it is the match, the team is the one being judged.
    fn eval_play(&self, play: &model::Play) -> Option<Judged> {
        self.eval_play_as(play, None)
    }

    /// [`Self::eval_play`] for a given team; `None` is the judged one. A
    /// library register picks its own: the side without the ball for an
    /// action without it, the side with it for a build-up.
    fn eval_play_as(&self, play: &model::Play, team: Option<u8>) -> Option<Judged> {
        let library = self.eval_library();
        let (key, _, action) = Self::eval_kind_of(&library, &play.tag)?;
        let clip = self.footage_at((play.start + play.end) / 2.0)?;
        let context = self.eval_context(&clip.id)?;
        let source = |at: f64| {
            let at = at.clamp(clip.start, clip.start + clip.duration);
            let seconds = clip.source_start + (at - clip.start) * clip.speed;
            (seconds.max(0.0) * 1000.0).round() as u64
        };
        let (start_ms, end_ms) = (source(play.start), source(play.end));
        let team = team.unwrap_or_else(|| {
            let library_clip = play
                .descriptors
                .iter()
                .any(|d| d == Self::LIBRARY_GOOD || d == Self::LIBRARY_BAD);
            if !library_clip {
                return self.eval_team;
            }
            // The file names the side judged: «… - rojo.mp4», «… equipo 2».
            if let Some(named) = self.library_team_named(&clip) {
                return named;
            }
            let mut held = [0usize; 3];
            for (ms, team) in &context.possession {
                if (start_ms..=end_ms).contains(ms)
                    && let Some(team @ 1..=2) = team
                {
                    held[usize::from(*team)] += 1;
                }
            }
            let (more, less) = if held[1] >= held[2] { (1, 2) } else { (2, 1) };
            // An action with the ball: the build-up, a coach's action
            // triggered by winning or holding it, or named as one.
            let with_ball = key == "build_up"
                || library.actions.iter().any(|def| {
                    def.key == key
                        && (matches!(
                            def.trigger,
                            concat_vision::football::learn::Trigger::BallWon
                                | concat_vision::football::learn::Trigger::OwnPossessionDeep { .. }
                        ) || ["ofensiv", "ataque", "salida", "posesión", "posesion"]
                            .iter()
                            .any(|word| def.name.to_lowercase().contains(word)))
                });
            if held[1] == held[2] {
                self.eval_team
            } else if with_ball {
                more
            } else {
                less
            }
        });
        self.eval_window(&library, &context, &key, action, start_ms, end_ms, team)
    }

    /// Evaluate: the actions of the judged team found in the footage - the
    /// four and the coach's own - each a register under its action's
    /// category, carrying its verdict as a descriptor and as stars. A run
    /// replaces the registers the last one made over the same footage.
    pub fn eval_run(&mut self) {
        use concat_vision::football::{eval::Verdict, learn};
        let Some(clip) = self
            .board_clip()
            .filter(|clip| clip.kind == model::ClipKind::Video)
        else {
            self.notify(&t("eval.noFootage"), true);
            return;
        };
        if !self.football_analysed(&clip.id) {
            self.notify(&t("eval.needsAnalysis"), true);
            return;
        }
        let Some(context) = self.eval_context(&clip.id) else {
            self.notify(&t("eval.needsCalibration"), true);
            return;
        };
        let library = self.eval_library();
        let end = clip.start + clip.duration;
        let timeline_at =
            |ms: u64| clip.start + (ms as f64 / 1000.0 - clip.source_start) / clip.speed.max(1e-6);

        // Every window to judge: the four actions' and the coach's own.
        let mut windows: Vec<(
            String,
            String,
            Option<concat_vision::football::eval::Action>,
            u64,
            u64,
        )> = context
            .occurrences()
            .into_iter()
            .filter(|occurrence| occurrence.team == self.eval_team)
            .map(|o| {
                (
                    o.action.key().to_string(),
                    o.action.name().to_string(),
                    Some(o.action),
                    o.start_ms,
                    o.end_ms,
                )
            })
            .collect();
        for def in &library.actions {
            for w in learn::find_windows(&context, def, self.eval_team) {
                windows.push((
                    def.key.clone(),
                    def.name.clone(),
                    None,
                    w.start_ms,
                    w.end_ms,
                ));
            }
        }

        // The categories, added once, each reaching as far as its window.
        let mut template = self.tag_template();
        let before = template.categories.len();
        let palette = [
            [0xe7, 0x4c, 0x3c],
            [0x34, 0x98, 0xdb],
            [0x2e, 0xcc, 0x71],
            [0x9b, 0x59, 0xb6],
            [0xe6, 0x7e, 0x22],
            [0x1a, 0xbc, 0x9c],
            [0xf1, 0xc4, 0x0f],
        ];
        let names: Vec<(String, f64)> = Self::EVAL_ACTIONS
            .iter()
            .zip([6.0, 10.0, 8.0, 15.0])
            .map(|(action, lag)| (action.name().to_string(), lag))
            .chain(
                library
                    .actions
                    .iter()
                    .map(|def| (def.name.clone(), def.window_s)),
            )
            .collect();
        for (index, (name, lag)) in names.into_iter().enumerate() {
            if !template
                .categories
                .iter()
                .any(|category| category.name == name)
            {
                template.categories.push(model::TagCategory {
                    name,
                    colour: palette[index % palette.len()],
                    lead: 0.0,
                    lag,
                });
            }
        }

        // Out with the last run's registers over this footage.
        let mut plays: Vec<model::Play> = self
            .timeline()
            .plays
            .iter()
            .filter(|play| {
                let ours = play.descriptors.iter().any(|d| d == Self::EVAL_AUTO)
                    && Self::eval_kind_of(&library, &play.tag).is_some();
                !(ours && play.start < end && play.end > clip.start)
            })
            .cloned()
            .collect();
        let mut number = plays
            .iter()
            .filter_map(|play| play.id.strip_prefix('p')?.parse::<u32>().ok())
            .max()
            .unwrap_or(0);
        let (mut good, mut improvable, mut bad, mut unknown) = (0, 0, 0, 0);
        for (key, name, action, start_ms, end_ms) in windows {
            let (start, stop) = (timeline_at(start_ms), timeline_at(end_ms).min(end));
            if start < clip.start || start >= end || stop <= start {
                continue;
            }
            let Some(judged) = self.eval_window(
                &library,
                &context,
                &key,
                action,
                start_ms,
                end_ms,
                self.eval_team,
            ) else {
                continue;
            };
            let verdict = judged.judgement.verdict;
            let rating = match verdict {
                Verdict::Good => {
                    good += 1;
                    5
                }
                Verdict::Improvable => {
                    improvable += 1;
                    3
                }
                Verdict::Bad => {
                    bad += 1;
                    1
                }
                Verdict::Unknown => {
                    unknown += 1;
                    0
                }
            };
            number += 1;
            let same = plays.iter().filter(|play| play.tag == name).count() + 1;
            plays.push(model::Play {
                id: format!("p{number}"),
                name: format!("{name} {same}"),
                tag: name,
                start,
                end: stop,
                descriptors: vec![Self::EVAL_AUTO.to_string(), verdict.label().to_string()],
                rating,
            });
        }
        plays.sort_by(|a, b| a.start.total_cmp(&b.start));
        let gesture = "evaluate";
        if template.categories.len() != before {
            self.apply_in(Some(gesture), Command::SetTagTemplate { template });
        }
        self.apply_in(Some(gesture), Command::SetPlays { plays });
        self.end_burst();
        let total = good + improvable + bad + unknown;
        self.notify(&tf("eval.done", &[&total, &good, &improvable, &bad]), false);
    }

    /// Writes one of the library's files.
    fn eval_write<T: serde::Serialize + ?Sized>(
        &self,
        name: &str,
        value: &T,
    ) -> Result<(), String> {
        let dir = self.eval_dir();
        std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
        let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
        std::fs::write(dir.join(name), bytes).map_err(|error| error.to_string())
    }

    /// The register in hand kept as a reference - the action done the way
    /// the team wants it, or the way it does not - and the models learned
    /// again from every reference kept.
    pub fn eval_reference(&mut self, good: bool) {
        use concat_vision::football::{eval::Reference, learn};
        let Some(play) = self.tag_in_hand() else {
            self.notify(&t("tag.tagFirst"), true);
            return;
        };
        let Some(judged) = self.eval_play(&play) else {
            self.notify(&t("eval.notAnAction"), true);
            return;
        };
        let library = self.eval_library();
        let mut references = library.references.clone();
        references.push(Reference {
            action_key: judged.key.clone(),
            good,
            measures: judged.measures,
        });
        let models = learn::retrain_all(&references);
        let written = self
            .eval_write("references.json", &references)
            .and_then(|()| self.eval_write("models.json", &models));
        if let Err(error) = written {
            self.notify(&error, true);
            return;
        }
        let of = |want: bool| {
            references
                .iter()
                .filter(|r| r.action_key == judged.key && r.good == want)
                .count()
        };
        let (goods, bads) = (of(true), of(false));
        let learned = match models.iter().find(|model| model.action_key == judged.key) {
            Some(model) => tf(
                "eval.learned",
                &[&format!("{:.0}", model.loo_accuracy.unwrap_or(0.0) * 100.0)],
            ),
            None => tf(
                "eval.needMore",
                &[&4usize.saturating_sub(goods), &4usize.saturating_sub(bads)],
            ),
        };
        self.notify(
            &format!(
                "{} · {learned}",
                tf("eval.referenceSaved", &[&goods, &bads])
            ),
            false,
        );
    }

    /// The profile in use tuned to the references of the register in hand's
    /// action: each threshold moved to where the good examples pass and the
    /// bad ones fail.
    pub fn eval_tune(&mut self) {
        use concat_vision::football::{eval::Profile, learn};
        let library = self.eval_library();
        let Some(key) = self
            .tag_in_hand()
            .and_then(|play| Self::eval_kind_of(&library, &play.tag))
            .map(|(key, _, _)| key)
        else {
            self.notify(&t("eval.notAnAction"), true);
            return;
        };
        let Some(mut profile) = self.eval_profile_in_use(&library) else {
            return;
        };
        let current = profile.rules.get(&key).cloned().unwrap_or_default();
        let suggestions = learn::suggest(&key, &library.references, &current);
        if suggestions.is_empty() {
            self.notify(&t("eval.tuneNeedsMore"), true);
            return;
        }
        learn::apply_suggestions(&mut profile, &key, &suggestions);
        let mut profiles = library.profiles.clone();
        let index = self.eval_profile.min(profiles.len().saturating_sub(1));
        if let Some(slot) = profiles.get_mut(index) {
            *slot = profile;
        }
        if let Err(error) = Profile::save_all(&self.eval_dir(), &profiles) {
            self.notify(&error, true);
            return;
        }
        let summary: Vec<String> = suggestions.iter().map(|s| s.rationale.clone()).collect();
        self.notify(
            &tf("eval.tuned", &[&summary.len(), &summary.join(" · ")]),
            false,
        );
    }

    /// The game-model profile picked.
    pub fn eval_profile_pick(&mut self, index: i32) {
        if let Ok(index) = usize::try_from(index) {
            self.eval_profile = index;
        }
    }

    /// The team picked, by its kit.
    pub fn eval_team_pick(&mut self, team: i32) {
        if let Ok(team @ 1..=2) = u8::try_from(team) {
            self.eval_team = team;
        }
    }

    /// The library's folder shown, with the profiles and the own actions
    /// written out first so there is a file to edit: thresholds, weights,
    /// which rules are major, and new actions with their triggers.
    pub fn eval_open_library(&mut self) {
        use concat_vision::football::{eval::Profile, learn};
        let dir = self.eval_dir();
        let library = self.eval_library();
        if !dir.join("profiles.json").exists()
            && let Err(error) = Profile::save_all(&dir, &library.profiles)
        {
            self.notify(&error, true);
            return;
        }
        if !dir.join("actions.json").exists() {
            // An example to copy from: a coach's action, judged on its own
            // generic KPIs.
            let example = vec![learn::ActionDef {
                key: "cierre_banda".into(),
                name: "Cierre de banda".into(),
                trigger: learn::Trigger::BallLost,
                window_s: 5.0,
                kpis: vec!["pressers_10m".into(), "nearest_ball".into(), "width".into()],
            }];
            if let Err(error) = learn::save_actions(&dir, &example) {
                self.notify(&error, true);
                return;
            }
        }
        if let Err(error) = crate::platform::reveal(&dir.join("profiles.json").to_string_lossy()) {
            self.notify(&error, true);
        }
    }

    /// The evaluation's part of the tagging pane.
    fn publish_evaluation(&self, editor: &Editor<'_>, models: &Models) {
        use concat_vision::football::eval::Verdict;
        let library = self.eval_library();
        sync(
            &models.eval_profiles,
            library
                .profiles
                .iter()
                .map(|profile| SharedString::from(profile.name.as_str()))
                .collect(),
        );
        editor.set_eval_profile(
            self.eval_profile
                .min(library.profiles.len().saturating_sub(1)) as i32,
        );
        editor.set_eval_team(i32::from(self.eval_team));
        let Some(judged) = self.tag_in_hand().and_then(|play| self.eval_play(&play)) else {
            editor.set_eval_kind(0);
            sync(&models.eval_reasons, Vec::new());
            return;
        };
        editor.set_eval_kind(match judged.judgement.verdict {
            Verdict::Good => 1,
            Verdict::Improvable => 2,
            Verdict::Bad => 3,
            Verdict::Unknown => 4,
        });
        editor.set_eval_verdict(judged.judgement.verdict.label().into());
        let of = |want: bool| {
            library
                .references
                .iter()
                .filter(|r| r.action_key == judged.key && r.good == want)
                .count()
        };
        editor.set_eval_refs(tf("eval.refs", &[&of(true), &of(false)]).into());
        sync(
            &models.eval_reasons,
            judged
                .judgement
                .reasons
                .iter()
                .map(|reason| SharedString::from(reason.as_str()))
                .collect(),
        );
    }

    // ── montage ──
    //
    // The cutting tools as a broadcast editor's keyboard has them, each one
    // a single key that does one thing to what is under the playhead: Q and
    // W take away what is left or right of it, closing the gap; I and O
    // mark a stretch of the timeline, E takes it out and closes up, R lifts
    // it and leaves the hole. Plays are marked stretches with a tag, kept
    // on the timeline, and the highlights are a new timeline cut from them
    // end to end.

    /// The clips Q and W act on: the selected ones under the playhead, or
    /// every clip under it on an unlocked lane when none of the selection
    /// is.
    fn under_playhead(&self) -> Vec<Clip> {
        let at = f64::from(self.playhead);
        let edge = f64::from(MIN_DURATION);
        let straddles =
            |clip: &Clip| clip.start + edge < at && at < clip.start + clip.duration - edge;
        let selected: Vec<Clip> = self
            .selection
            .iter()
            .filter_map(|id| self.clip(id).cloned())
            .filter(|clip| straddles(clip) && !self.locked(&clip.track_id))
            .collect();
        if !selected.is_empty() {
            return selected;
        }
        self.timeline()
            .clips
            .iter()
            .filter(|clip| straddles(clip) && !self.locked(&clip.track_id))
            .map(|clip| (**clip).clone())
            .collect()
    }

    /// Q: what is left of the playhead goes, and the lane closes up behind
    /// it, so the playhead lands on the cut.
    pub fn delete_left(&mut self) {
        let clips = self.under_playhead();
        if clips.is_empty() {
            self.notify(&t("montage.nothingUnderPlayhead"), true);
            return;
        }
        let at = f64::from(self.playhead);
        let commands: Vec<Command> = clips
            .iter()
            .map(|clip| Command::TrimClip {
                clip_id: clip.id.clone(),
                edge: TrimEdge::Start,
                delta: at - clip.start,
                ripple: true,
            })
            .collect();
        let back = clips.iter().map(|clip| clip.start).fold(f64::MAX, f64::min);
        self.apply(Command::Batch { commands });
        self.seek(back as f32);
    }

    /// W: what is right of the playhead goes, and the lane closes up.
    pub fn delete_right(&mut self) {
        let clips = self.under_playhead();
        if clips.is_empty() {
            self.notify(&t("montage.nothingUnderPlayhead"), true);
            return;
        }
        let at = f64::from(self.playhead);
        let commands: Vec<Command> = clips
            .iter()
            .map(|clip| Command::TrimClip {
                clip_id: clip.id.clone(),
                edge: TrimEdge::End,
                delta: at - (clip.start + clip.duration),
                ripple: true,
            })
            .collect();
        self.apply(Command::Batch { commands });
    }

    /// I: the stretch begins at the playhead.
    pub fn mark_in_here(&mut self) {
        self.mark_in = Some(self.playhead);
        if self.mark_out.is_some_and(|out| out <= self.playhead) {
            self.mark_out = None;
        }
    }

    /// O: the stretch ends at the playhead.
    pub fn mark_out_here(&mut self) {
        self.mark_out = Some(self.playhead);
        if self.mark_in.is_some_and(|start| start >= self.playhead) {
            self.mark_in = None;
        }
    }

    /// X: no stretch.
    pub fn clear_marks(&mut self) {
        self.mark_in = None;
        self.mark_out = None;
    }

    /// The marked stretch, in timeline seconds: from the in mark, or the
    /// start, to the out mark, or the end. `None` with no mark at all.
    fn marked(&self) -> Option<(f64, f64)> {
        if self.mark_in.is_none() && self.mark_out.is_none() {
            return None;
        }
        let start = f64::from(self.mark_in.unwrap_or(0.0));
        let end = f64::from(self.mark_out.unwrap_or(self.duration()));
        (end - start > f64::from(MIN_DURATION)).then_some((start, end))
    }

    /// E (`closing`) or R: the marked stretch taken out of every unlocked
    /// lane. E closes the timeline up behind it, the way Extract does in
    /// every editor; R lifts it and leaves the hole. One undo step.
    pub fn take_out_marked(&mut self, closing: bool) {
        let Some((start, end)) = self.marked() else {
            self.notify(&t("montage.markFirst"), true);
            return;
        };
        let gesture = if closing { "extract" } else { "lift" };
        let edge = f64::from(MIN_DURATION);
        for cut in [start, end] {
            let ids: Vec<String> = self
                .timeline()
                .clips
                .iter()
                .filter(|clip| {
                    clip.start + edge < cut
                        && cut < clip.start + clip.duration - edge
                        && !self.locked(&clip.track_id)
                })
                .map(|clip| clip.id.clone())
                .collect();
            if !ids.is_empty() {
                self.apply_in(
                    Some(gesture),
                    Command::SplitClips {
                        clip_ids: ids,
                        time: cut,
                    },
                );
            }
        }
        let inside: Vec<String> = self
            .timeline()
            .clips
            .iter()
            .filter(|clip| {
                clip.start >= start - edge
                    && clip.start + clip.duration <= end + edge
                    && !self.locked(&clip.track_id)
            })
            .map(|clip| clip.id.clone())
            .collect();
        if !inside.is_empty() {
            self.apply_in(
                Some(gesture),
                Command::RemoveClips {
                    clip_ids: inside,
                    ripple: false,
                },
            );
        }
        if closing {
            let length = end - start;
            let moves: Vec<ClipMove> = self
                .timeline()
                .clips
                .iter()
                .filter(|clip| clip.start >= end - edge && !self.locked(&clip.track_id))
                .map(|clip| ClipMove {
                    clip_id: clip.id.clone(),
                    start: (clip.start - length).max(0.0),
                    track_id: clip.track_id.clone(),
                })
                .collect();
            if !moves.is_empty() {
                self.apply_in(Some(gesture), Command::MoveClips { moves });
            }
        }
        self.end_burst();
        self.selection.clear();
        self.clear_marks();
        self.seek(start as f32);
    }

    /// The tags a play can wear, in the menu's order, with the colour each
    /// is drawn in on the ruler.
    pub const PLAY_TAGS: [(&'static str, [u8; 3]); 8] = [
        ("goal", [0x2e, 0xcc, 0x71]),
        ("chance", [0xf1, 0xc4, 0x0f]),
        ("press", [0xe6, 0x7e, 0x22]),
        ("attack", [0x34, 0x98, 0xdb]),
        ("defence", [0x9b, 0x59, 0xb6]),
        ("setPiece", [0x1a, 0xbc, 0x9c]),
        ("error", [0xe7, 0x4c, 0x3c]),
        ("other", [0x95, 0xa5, 0xa6]),
    ];

    /// Seconds as minutes and seconds, the way a match clock reads.
    fn montage_clock(seconds: f32) -> String {
        let whole = seconds.max(0.0).round() as u32;
        format!("{}:{:02}", whole / 60, whole % 60)
    }

    /// A tag's name in the interface's language.
    fn play_tag_name(tag: &str) -> String {
        if Self::PLAY_TAGS.iter().any(|(known, _)| *known == tag) {
            t(&format!("montage.tag.{tag}"))
        } else {
            tag.to_owned()
        }
    }

    /// The plays on the active timeline, by where they start.
    fn plays_in_order(&self) -> Vec<model::Play> {
        let mut plays = self.timeline().plays.clone();
        plays.sort_by(|a, b| a.start.total_cmp(&b.start));
        plays
    }

    /// P, or a tag from the menu: the marked stretch becomes a play - or,
    /// with nothing marked, the selected clip's span, or failing that the
    /// clip under the playhead's.
    pub fn add_play(&mut self, tag: &str) {
        let span = self.marked().or_else(|| {
            let clip = self
                .sole_selection()
                .and_then(|id| self.clip(&id).cloned())
                .or_else(|| self.under_playhead().into_iter().next())?;
            Some((clip.start, clip.start + clip.duration))
        });
        let Some((start, end)) = span else {
            self.notify(&t("montage.markFirst"), true);
            return;
        };
        let mut plays = self.timeline().plays.clone();
        let number = plays
            .iter()
            .filter_map(|play| play.id.strip_prefix('p')?.parse::<u32>().ok())
            .max()
            .unwrap_or(0)
            + 1;
        let same = plays.iter().filter(|play| play.tag == tag).count() + 1;
        plays.push(model::Play {
            id: format!("p{number}"),
            name: format!("{} {same}", Self::play_tag_name(tag)),
            tag: tag.to_owned(),
            start,
            end,
            descriptors: Vec::new(),
            rating: 0,
        });
        self.apply(Command::SetPlays { plays });
        self.clear_marks();
        self.notify(
            &tf(
                "montage.playAdded",
                &[
                    &Self::play_tag_name(tag),
                    &Self::montage_clock(start as f32),
                ],
            ),
            false,
        );
    }

    /// Deletes the play under the playhead, the latest-starting one when
    /// plays overlap there.
    pub fn remove_play_here(&mut self) {
        let at = f64::from(self.playhead);
        let mut plays = self.timeline().plays.clone();
        let Some(index) = plays
            .iter()
            .enumerate()
            .filter(|(_, play)| play.start <= at && at <= play.end)
            .max_by(|a, b| a.1.start.total_cmp(&b.1.start))
            .map(|(index, _)| index)
        else {
            self.notify(&t("montage.noPlayHere"), true);
            return;
        };
        plays.remove(index);
        self.apply(Command::SetPlays { plays });
    }

    /// [ and ]: the playhead to the start of the play before or after it.
    pub fn jump_play(&mut self, forward: bool) {
        let at = f64::from(self.playhead);
        let plays = self.plays_in_order();
        let target = if forward {
            plays.iter().find(|play| play.start > at + 1e-3)
        } else {
            plays.iter().rev().find(|play| play.start < at - 1e-3)
        };
        if let Some(play) = target {
            let (start, end) = (play.start, play.end);
            self.seek(start as f32);
            self.mark_in = Some(start as f32);
            self.mark_out = Some(end as f32);
        }
    }

    /// The highlights: a new timeline holding every play - or every play
    /// of `tag` - cut from this one end to end, in the order they happen,
    /// on the lanes they were on. The source timeline is left as it was.
    pub fn build_highlights(&mut self, tag: Option<&str>) {
        let plays: Vec<model::Play> = self
            .plays_in_order()
            .into_iter()
            .filter(|play| tag.is_none_or(|tag| play.tag == tag))
            .collect();
        let name = match tag {
            Some(tag) => tf("montage.highlightsOf", &[&Self::play_tag_name(tag)]),
            None => t("montage.highlights"),
        };
        self.build_highlights_from(plays, name);
    }

    /// A new timeline called `name` holding `plays`, cut from this one end
    /// to end in the order they happen; see `build_highlights`.
    pub fn build_highlights_from(&mut self, mut plays: Vec<model::Play>, name: String) {
        plays.sort_by(|a, b| a.start.total_cmp(&b.start));
        if plays.is_empty() {
            self.notify(&t("montage.noPlays"), true);
            return;
        }
        let source = self.timeline().clone();
        let gesture = "highlights";
        self.apply_in(Some(gesture), Command::AddTimeline);
        let id = self.timeline().id.clone();
        self.apply_in(
            Some(gesture),
            Command::RenameTimeline {
                timeline_id: id,
                name,
            },
        );
        while self.timeline().tracks.len() < source.tracks.len() {
            self.apply_in(Some(gesture), Command::AddTrack { bottom: false });
        }
        let lanes: Vec<String> = self
            .timeline()
            .tracks
            .iter()
            .map(|track| track.id.clone())
            .collect();
        let edge = f64::from(MIN_DURATION);
        let mut cursor = 0.0;
        for play in &plays {
            let mut pieces: Vec<&Arc<Clip>> = source
                .clips
                .iter()
                .filter(|clip| {
                    clip.start < play.end - edge && clip.start + clip.duration > play.start + edge
                })
                .collect();
            pieces.sort_by(|a, b| a.start.total_cmp(&b.start));
            for clip in pieces {
                let Some(row) = source
                    .tracks
                    .iter()
                    .position(|track| track.id == clip.track_id)
                else {
                    continue;
                };
                let from = clip.start.max(play.start);
                let to = (clip.start + clip.duration).min(play.end);
                let mut piece = (**clip).clone();
                piece.source_start = clip.source_start + (from - clip.start) * clip.speed;
                piece.start = cursor + (from - play.start);
                piece.duration = to - from;
                self.place_piece(&piece, &lanes[row], gesture);
            }
            cursor += play.end - play.start;
        }
        self.end_burst();
        self.selection.clear();
        self.seek(0.0);
        self.notify(&tf("montage.highlightsBuilt", &[&plays.len()]), false);
    }

    /// One stretch of a clip laid on `lane` exactly as `piece` says - its
    /// start, length and in-point - with the clip's looks. Titles and
    /// shapes have no in-point and are placed as copies are.
    fn place_piece(&mut self, piece: &Clip, lane: &str, gesture: &str) {
        if matches!(piece.kind, model::ClipKind::Text | model::ClipKind::Shape) {
            self.place_copy(piece, lane, piece.start, Some(gesture));
            return;
        }
        let Some(created) = self.apply_in(
            Some(gesture),
            Command::AddClip {
                media_id: piece.media_id.clone(),
                track_id: lane.to_owned(),
                start: piece.start,
                ripple: false,
            },
        ) else {
            return;
        };
        let mut commands = Vec::new();
        if (piece.speed - 1.0).abs() > 1e-9 {
            commands.push(Command::SetClipSpeed {
                clip_id: created.clone(),
                speed: piece.speed,
            });
        }
        self.apply_in(Some(gesture), Command::Batch { commands });
        let Some(placed) = self.clip(&created).cloned() else {
            return;
        };
        // The head cut to the in-point, which moves the clip right by as
        // much; the tail cut to the length; and the clip back where it goes.
        let head = (piece.source_start - placed.source_start) / piece.speed.max(1e-6);
        let mut commands = vec![Command::TrimClip {
            clip_id: created.clone(),
            edge: TrimEdge::Start,
            delta: head,
            ripple: false,
        }];
        let tail = piece.duration - (placed.duration - head);
        commands.push(Command::TrimClip {
            clip_id: created.clone(),
            edge: TrimEdge::End,
            delta: tail,
            ripple: false,
        });
        commands.push(Command::MoveClips {
            moves: vec![ClipMove {
                clip_id: created.clone(),
                start: piece.start,
                track_id: lane.to_owned(),
            }],
        });
        commands.push(Command::SetClipTransform {
            clip_id: created.clone(),
            scale: Some(piece.scale),
            offset_x: Some(piece.offset_x),
            offset_y: Some(piece.offset_y),
            rotation: Some(piece.rotation),
            stretch_x: Some(piece.stretch_x),
            stretch_y: Some(piece.stretch_y),
        });
        commands.push(Command::UpdateClip {
            clip_id: created.clone(),
            patch: ClipPatch {
                name: Some(piece.name.clone()),
                volume: Some(piece.volume),
                opacity: Some(piece.opacity),
                preserve_pitch: Some(piece.preserve_pitch),
                muted: piece.muted,
                flip_h: Some(piece.flip_h),
                flip_v: Some(piece.flip_v),
                blend: Some(piece.blend.clone()),
                crop: Some(piece.crop),
                filters: Some(piece.filters.clone()),
                video_effects: Some(piece.video_effects.clone()),
                ..ClipPatch::default()
            },
        });
        if !piece.football.is_empty() {
            commands.push(Command::SetClipFootball {
                clip_id: created.clone(),
                marks: piece.football.clone(),
            });
        }
        self.apply_in(Some(gesture), Command::Batch { commands });
    }

    /// The Montage menu's rows and the keys' actions, by id.
    pub fn montage_action(&mut self, action: &str) {
        match action {
            "delete-left" => self.delete_left(),
            "delete-right" => self.delete_right(),
            "mark-in" => self.mark_in_here(),
            "mark-out" => self.mark_out_here(),
            "clear-marks" => self.clear_marks(),
            "extract" => self.take_out_marked(true),
            "lift" => self.take_out_marked(false),
            "play-add" => self.add_play("other"),
            "play-remove" => self.remove_play_here(),
            "play-prev" => self.jump_play(false),
            "play-next" => self.jump_play(true),
            "highlights" => self.build_highlights(None),
            "workspace-analysis" => self.toggle_analysis_workspace(),
            "pointer-tool" => self.analysis_tool_pick(0),
            "back-five" => {
                let at = (self.playhead - 5.0).max(0.0);
                self.seek(at);
            }
            "forward-five" => {
                let at = (self.playhead + 5.0).min(self.duration());
                self.seek(at);
            }
            other => {
                if let Some(tag) = other.strip_prefix("play-add:") {
                    self.add_play(tag);
                } else if let Some(tag) = other.strip_prefix("highlights:") {
                    self.build_highlights(Some(tag));
                } else if let Some(id) = other.strip_prefix("play-go:") {
                    if let Some(play) = self.timeline().plays.iter().find(|play| play.id == id) {
                        let (start, end) = (play.start, play.end);
                        self.seek(start as f32);
                        self.mark_in = Some(start as f32);
                        self.mark_out = Some(end as f32);
                    }
                }
            }
        }
    }

    /// The Montage menu: the cutting keys, the marks, and the plays.
    fn montage_menu(&self) -> Vec<MenuItemData> {
        let row =
            |id: &str, label: String, glyph: Glyph, shortcut: &str, enabled: bool| MenuItemData {
                id: id.into(),
                label: label.into(),
                kind: MenuRow::Action,
                glyph,
                shortcut: shortcut.into(),
                enabled,
                danger: false,
                checkable: false,
                checked: false,
            };
        let rule = || MenuItemData {
            kind: MenuRow::Separator,
            ..Default::default()
        };
        let under = !self.under_playhead().is_empty();
        let marked = self.marked().is_some();
        let plays = self.plays_in_order();
        let mut rows = vec![
            row(
                "delete-left",
                t("montage.deleteLeft"),
                Glyph::Razor,
                "Q",
                under,
            ),
            row(
                "delete-right",
                t("montage.deleteRight"),
                Glyph::Razor,
                "W",
                under,
            ),
            rule(),
            row("mark-in", t("montage.markIn"), Glyph::None, "I", true),
            row("mark-out", t("montage.markOut"), Glyph::None, "O", true),
            row(
                "clear-marks",
                t("montage.clearMarks"),
                Glyph::None,
                "X",
                marked,
            ),
            row("extract", t("montage.extract"), Glyph::Trash, "E", marked),
            row("lift", t("montage.lift"), Glyph::Trash, "R", marked),
            rule(),
        ];
        for (tag, _) in Self::PLAY_TAGS {
            rows.push(row(
                &format!("play-add:{tag}"),
                tf("montage.addPlayAs", &[&Self::play_tag_name(tag)]),
                Glyph::Plus,
                if tag == "other" { "P" } else { "" },
                true,
            ));
        }
        rows.push(rule());
        rows.push(row(
            "play-prev",
            t("montage.prevPlay"),
            Glyph::SkipBack,
            "[",
            !plays.is_empty(),
        ));
        rows.push(row(
            "play-next",
            t("montage.nextPlay"),
            Glyph::SkipForward,
            "]",
            !plays.is_empty(),
        ));
        let at = f64::from(self.playhead);
        rows.push(row(
            "play-remove",
            t("montage.removePlay"),
            Glyph::Trash,
            "",
            plays.iter().any(|play| play.start <= at && at <= play.end),
        ));
        rows.push(rule());
        rows.push(row(
            "highlights",
            tf("montage.buildHighlights", &[&plays.len()]),
            Glyph::None,
            "",
            !plays.is_empty(),
        ));
        for (tag, _) in Self::PLAY_TAGS {
            let count = plays.iter().filter(|play| play.tag == tag).count();
            if count > 0 {
                rows.push(row(
                    &format!("highlights:{tag}"),
                    tf(
                        "montage.buildHighlightsOf",
                        &[&Self::play_tag_name(tag), &count],
                    ),
                    Glyph::None,
                    "",
                    true,
                ));
            }
        }
        if !plays.is_empty() {
            rows.push(rule());
            for play in plays.iter().take(20) {
                rows.push(row(
                    &format!("play-go:{}", play.id),
                    format!(
                        "{}  {}  ({})",
                        Self::montage_clock(play.start as f32),
                        play.name,
                        Self::montage_clock((play.end - play.start) as f32)
                    ),
                    Glyph::None,
                    "",
                    true,
                ));
            }
        }
        rows
    }

    /// The plays as the ruler draws them, and the marked stretch.
    fn montage_marks(&self) -> Vec<PlayMarkData> {
        let template = self.tag_template();
        self.plays_in_order()
            .into_iter()
            .map(|play| {
                let [r, g, b] = self.play_colour(&template, &play.tag);
                PlayMarkData {
                    start: play.start as f32,
                    end: play.end as f32,
                    label: play.name.clone().into(),
                    colour: slint::Color::from_rgb_u8(r, g, b),
                }
            })
            .collect()
    }

    // ── football ──
    //
    // The football analysis finds the players and the ball in a clip's
    // footage and follows them; a mark on a tracked thing shades it,
    // spotlights it, rings it, takes it out of the picture or keeps it
    // alone. Moving a player is two marks on two layers: a copy of the
    // clip above that keeps only him, which the stage moves like any
    // picture, and the player erased from the clip beneath. Copying one is
    // the first half, and the copy cuts, copies and pastes like any clip.

    /// The analysis request for clip `id`, when the project has a folder.
    fn football_request(&self, id: &str) -> Option<concat_host::FootballRequest> {
        let session = self.session.as_ref()?;
        let project = std::path::PathBuf::from(session.path());
        concat_host::FootballRequest::for_clip(self.project(), &project, id)
    }

    /// Whether clip `id`'s footage has been analysed.
    pub fn football_analysed(&self, id: &str) -> bool {
        self.football_request(id)
            .is_some_and(|request| request.analysed())
    }

    /// Finds and tracks the players and the ball in the selected clip's
    /// footage, unless a football job is running.
    pub fn football_analyse(&mut self) {
        if self.football_job.is_some() || self.host.footballs.is_busy() {
            return;
        }
        let Some(id) = self.sole_selection() else {
            return;
        };
        let Some(request) = self.football_request(&id) else {
            self.notify(&t("football.needsFootage"), true);
            return;
        };
        self.football_job = Some(("analyse".to_owned(), id.clone(), 0.0));
        self.football_chosen = None;
        // The instant being looked at is analysed first; see
        // `Footballs::analyse`.
        let focus = self.clip(&id).map(|clip| self.source_at_playhead(clip));
        let footballs = Arc::clone(&self.host.footballs);
        let epoch = crate::host::project_epoch();
        spawn_in_project(
            move || {
                let mut last = -1.0f32;
                let mut on_window = move || {
                    on_ui_in_project(epoch, |studio, _, _| {
                        studio.land_pending_mark();
                        studio.request_preview();
                    });
                };
                footballs.analyse(
                    &request,
                    focus,
                    &mut |progress| {
                        let now = match progress {
                            concat_host::cutout::Progress::Fetching { .. } => 0.0,
                            concat_host::cutout::Progress::Analysing(fraction) => fraction,
                        };
                        if now - last >= 0.01 {
                            last = now;
                            on_ui_in_project(epoch, move |studio, _, _| {
                                if let Some(job) = studio.football_job.as_mut() {
                                    job.2 = now;
                                }
                            });
                        }
                    },
                    &mut on_window,
                )
            },
            |studio, _, _, result| {
                studio.football_job = None;
                match result {
                    Ok(_) => {
                        studio.football_picking = true;
                        studio.painting = false;
                        // The silhouettes were read in the old boxes and are
                        // gone; every marked player is read again.
                        studio.football_tried.clear();
                        studio.notify(&t("football.analysed"), false);
                        studio.land_pending_mark();
                        studio.request_preview();
                        studio.ensure_football_silhouettes();
                        studio.library_next();
                    }
                    Err(error) => {
                        // The press kept for it would land on whatever is
                        // analysed next, somewhere else.
                        studio.football_pending = None;
                        if !error.contains("cancelled") {
                            studio.notify(&tf("football.failed", &[&error]), true);
                        }
                        studio.library_next();
                    }
                }
            },
        );
    }

    /// Turns picking on the stage on or off.
    pub fn football_pick(&mut self, on: bool) {
        self.football_picking = on;
        if on {
            self.painting = false;
        }
        self.request_preview();
    }

    /// A press on the stage while picking: the tracked thing under it in
    /// the selected clip becomes the chosen one. `false` when picking is
    /// off or the press is not on the selected clip, so the press does
    /// what it would have.
    fn football_pick_at(&mut self, x: f32, y: f32) -> bool {
        if !self.football_picking {
            return false;
        }
        let Some(id) = self.sole_selection() else {
            return false;
        };
        let Some(clip) = self.clip(&id).cloned() else {
            return false;
        };
        if !self
            .stage_clips()
            .into_iter()
            .any(|shown| shown.id == clip.id)
        {
            return false;
        }
        let Some(request) = self.football_request(&id) else {
            return false;
        };
        let Some(set) = concat_vision::football::shared(&request.dir()) else {
            return false;
        };
        let [u, v] = self.stage_to_source(&clip, f64::from(x), f64::from(y));
        let at = self.source_at_playhead(&clip);
        self.football_chosen = set.pick(at, u as f32, v as f32).map(|tracked| tracked.id);
        self.request_preview();
        true
    }

    /// Puts a mark of `style` (0 shade, 1 spotlight, 2 ring, 3 erase,
    /// 4 isolate) on the chosen track of the selected clip, or takes it
    /// off when it is there already.
    pub fn football_mark(&mut self, style: i32) {
        let style = match style {
            0 => model::MarkStyle::Shade,
            1 => model::MarkStyle::Spotlight,
            2 => model::MarkStyle::Ring,
            3 => model::MarkStyle::Erase,
            4 => model::MarkStyle::Isolate,
            5 => model::MarkStyle::Arrow,
            7 => model::MarkStyle::Trail,
            8 => model::MarkStyle::Lens,
            9 => model::MarkStyle::Distance,
            10 => model::MarkStyle::Speed,
            11 => model::MarkStyle::LastLine,
            12 => model::MarkStyle::Lanes,
            _ => model::MarkStyle::Frame,
        };
        let (Some(id), Some(track)) = (self.sole_selection(), self.football_chosen) else {
            self.notify(&t("football.pickFirst"), true);
            return;
        };
        let Some(clip) = self.clip(&id).cloned() else {
            return;
        };
        let mut marks = clip.football.clone();
        match marks
            .iter()
            .position(|mark| mark.track == track && mark.style == style)
        {
            Some(index) => {
                marks.remove(index);
            }
            None => marks.push(model::FootballMark::new(track, style)),
        }
        self.apply(Command::SetClipFootball {
            clip_id: id.clone(),
            marks,
        });
        if matches!(
            style,
            model::MarkStyle::Shade | model::MarkStyle::Erase | model::MarkStyle::Isolate
        ) {
            self.ensure_football_silhouettes();
        }
    }

    /// Takes the chosen player out of the selected clip onto a layer of his
    /// own above it: a copy of the clip keeping only him, selected so the
    /// stage moves him. `moving` also erases him from the clip beneath,
    /// which is moving him; without it he is copied.
    pub fn football_extract(&mut self, moving: bool) {
        let (Some(id), Some(track)) = (self.sole_selection(), self.football_chosen) else {
            self.notify(&t("football.pickFirst"), true);
            return;
        };
        let Some(source) = self.clip(&id).cloned() else {
            return;
        };
        if !source.kind.is_visual() || self.locked(&source.track_id) {
            return;
        }
        let gesture = if moving {
            "football-move"
        } else {
            "football-copy"
        };
        let Some(lane) = self.apply_in(Some(gesture), Command::AddTrack { bottom: false }) else {
            return;
        };
        let Some(created) = self.place_copy(&source, &lane, source.start, Some(gesture)) else {
            return;
        };
        // The copy carries none of the source's other marks: it is the
        // player and nothing else.
        self.apply_in(
            Some(gesture),
            Command::SetClipFootball {
                clip_id: created.clone(),
                marks: vec![model::FootballMark::new(track, model::MarkStyle::Isolate)],
            },
        );
        if moving {
            let mut marks = source.football.clone();
            if !marks
                .iter()
                .any(|mark| mark.track == track && mark.style == model::MarkStyle::Erase)
            {
                marks.push(model::FootballMark::new(track, model::MarkStyle::Erase));
            }
            self.apply_in(
                Some(gesture),
                Command::SetClipFootball {
                    clip_id: id.clone(),
                    marks,
                },
            );
        }
        self.end_burst();
        self.selection = vec![created];
        self.football_picking = false;
        self.notify(
            &t(if moving {
                "football.movedHint"
            } else {
                "football.copiedHint"
            }),
            false,
        );
        self.ensure_football_silhouettes();
    }

    /// The colours a mark can wear, as the inspector's swatches show them:
    /// the accent, then a red, a blue, a yellow, white and an orange - two
    /// teams' worth of kit and something for the ball.
    const MARK_COLOURS: [[u8; 3]; 6] = [
        [0xcb, 0xf5, 0x3f],
        [0xe7, 0x4c, 0x3c],
        [0x3f, 0xa9, 0xf5],
        [0xf5, 0xd6, 0x3f],
        [0xff, 0xff, 0xff],
        [0xff, 0x8c, 0x1a],
    ];

    /// The chosen player's marks on the selected clip in the palette's
    /// colour `index`, as one undo step.
    pub fn football_colour(&mut self, index: i32) {
        let (Some(id), Some(track)) = (self.sole_selection(), self.football_chosen) else {
            return;
        };
        let Some(colour) = usize::try_from(index)
            .ok()
            .and_then(|index| Self::MARK_COLOURS.get(index).copied())
        else {
            return;
        };
        let Some(clip) = self.clip(&id).cloned() else {
            return;
        };
        let marks: Vec<model::FootballMark> = clip
            .football
            .iter()
            .cloned()
            .map(|mark| {
                if mark.track == track {
                    model::FootballMark { colour, ..mark }
                } else {
                    mark
                }
            })
            .collect();
        self.apply(Command::SetClipFootball { clip_id: id, marks });
    }

    /// One knob of the chosen player's marks on the selected clip - 0 size,
    /// 1 thickness, 2 length, 3 strength, 4 width - set on all of them at once. A
    /// drag on the knob is one undo step, as every inspector knob is.
    pub fn football_param(&mut self, which: i32, value: f32) {
        let (Some(id), Some(track)) = (self.sole_selection(), self.football_chosen) else {
            return;
        };
        let Some(clip) = self.clip(&id).cloned() else {
            return;
        };
        let value = f64::from(value);
        let marks: Vec<model::FootballMark> = clip
            .football
            .iter()
            .cloned()
            .map(|mut mark| {
                if mark.track == track {
                    match which {
                        0 => mark.size = value,
                        1 => mark.thickness = value,
                        2 => mark.squash = value,
                        4 => mark.width = value,
                        _ => mark.strength = value,
                    }
                }
                mark
            })
            .collect();
        if marks == clip.football {
            return;
        }
        let gesture = format!("football-knob-{which}");
        self.apply_in(
            Some(&gesture),
            Command::SetClipFootball { clip_id: id, marks },
        );
    }

    /// Takes every football mark off the selected clip.
    pub fn football_clear(&mut self) {
        let Some(id) = self.sole_selection() else {
            return;
        };
        if self.clip(&id).is_some_and(|clip| !clip.football.is_empty()) {
            self.apply(Command::SetClipFootball {
                clip_id: id,
                marks: Vec::new(),
            });
        }
    }

    /// Starts reading the next silhouette a mark on the active timeline
    /// needs and does not have, unless a football job is running. Called
    /// after every change and again when a job finishes, so a mark put on
    /// while another player's silhouette was being read gets its turn - the
    /// way `ensure_cutouts` queues the cutout analyses. Each player is tried
    /// once per analysis: one the brush model cannot read stays on its box
    /// rather than being read over and over.
    pub fn ensure_football_silhouettes(&mut self) {
        if self.football_job.is_some() || self.host.footballs.is_busy() {
            return;
        }
        let needs = |style: model::MarkStyle| {
            matches!(
                style,
                model::MarkStyle::Shade | model::MarkStyle::Erase | model::MarkStyle::Isolate
            )
        };
        let marked: Vec<(String, Vec<u32>)> = self
            .timeline()
            .clips
            .iter()
            .filter(|clip| clip.football.iter().any(|mark| needs(mark.style)))
            .map(|clip| {
                let tracks = clip
                    .football
                    .iter()
                    .filter(|mark| needs(mark.style))
                    .map(|mark| mark.track)
                    .collect();
                (clip.id.clone(), tracks)
            })
            .collect();
        for (id, tracks) in marked {
            let Some(request) = self.football_request(&id) else {
                continue;
            };
            if !request.analysed() {
                continue;
            }
            let dir = request.dir();
            let wanted = tracks.into_iter().find(|track| {
                !self
                    .football_tried
                    .contains(&(request.media_path.clone(), *track))
                    && concat_vision::football::silhouettes(&dir, *track).is_empty()
            });
            if let Some(track) = wanted {
                self.football_tried
                    .insert((request.media_path.clone(), track));
                self.football_silhouettes(id, track);
                return;
            }
        }
    }

    /// Reads track `track`'s silhouettes in clip `id`'s footage, unless a
    /// football job is running; the marks use its box until they arrive.
    fn football_silhouettes(&mut self, id: String, track: u32) {
        if self.football_job.is_some() || self.host.footballs.is_busy() {
            return;
        }
        let Some(request) = self.football_request(&id) else {
            return;
        };
        self.football_job = Some(("silhouettes".to_owned(), id, 0.0));
        let footballs = Arc::clone(&self.host.footballs);
        let epoch = crate::host::project_epoch();
        spawn_in_project(
            move || {
                let mut last = -1.0f32;
                footballs.silhouettes(&request, track, &mut |progress| {
                    let now = match progress {
                        concat_host::cutout::Progress::Fetching { received, total } => {
                            received as f32 / total.max(1) as f32
                        }
                        concat_host::cutout::Progress::Analysing(fraction) => fraction,
                    };
                    if (now - last).abs() >= 0.01 {
                        last = now;
                        on_ui_in_project(epoch, move |studio, _, _| {
                            if let Some(job) = studio.football_job.as_mut() {
                                job.2 = now;
                            }
                        });
                    }
                })
            },
            |studio, _, _, result| {
                studio.football_job = None;
                match result {
                    Ok(_) => studio.request_preview(),
                    Err(error) if error.contains("cancelled") => {}
                    Err(error) => studio.notify(&tf("football.failed", &[&error]), true),
                }
                studio.ensure_football_silhouettes();
            },
        );
    }

    // ── enhance ──
    //
    // The restoration model reads a clip's file frame by frame and writes
    // an enhanced copy into the project's cache; the clip is then pointed
    // at the copy, as one undo step, and shows it wherever it showed the
    // original. The window's part is to start the job, keep the readout
    // moving, and re-point the clip when the copy is whole.

    /// Enhances the media of clip `id`: starts the job, unless one is
    /// running, and says so.
    /// Writes one clip's sound as it plays - trimmed, at its speed, with its
    /// level, fades and effects baked in - to a WAV in the project's audio
    /// folder, and puts the file in the bin under Generated › Processed. A
    /// copy, not a replacement: the clip keeps its media and its settings,
    /// and the file is independent of both, so it can be cut, moved and
    /// exported like any import.
    ///
    /// The export's own audio path does the work, on the clip alone: the
    /// project is flattened with just this clip on its timeline, the one
    /// export clip that comes out is moved to the origin, and the mixer
    /// writes it the way it writes an export's soundtrack.
    /// The clip's menu: its sound as it plays - trimmed, at its speed, with
    /// its level, fades and effects baked in - written to a WAV the user
    /// names, and brought into the bin as a file of its own, selected,
    /// with the bin showing all its media so the new file is in view.
    ///
    /// A plain import, not a Generated › Processed one: the user picked
    /// where the file goes and what it is called, which makes it theirs
    /// the way a file dragged in is. On a phone, which has no save dialog,
    /// it goes in the project's audio folder under the clip's name.
    pub fn export_clip_audio(&mut self, id: &str) {
        let Some(clip) = self.clip(id).cloned() else {
            return;
        };
        if !self.clip_has_sound(&clip) {
            self.notify(&t("studio.clipHasNoSound"), true);
            return;
        }
        let Some(project_dir) = self
            .session
            .as_ref()
            .map(|session| std::path::PathBuf::from(session.path()))
        else {
            return;
        };
        let mut alone = self.project().clone();
        alone.active_mut().clips.retain(|held| held.id == clip.id);
        let Some(mut piece) =
            concat_export::flatten::flatten_timeline_in(&alone, None, Some(&project_dir))
                .into_iter()
                .next()
        else {
            self.notify(&t("studio.clipHasNoSound"), true);
            return;
        };
        piece.start = 0.0;
        piece.muted = false;
        let duration = piece.duration;
        let pieces = concat_export::audio_pieces(&piece);

        // Where it goes: the user's choice, offered under the clip's name
        // in the project's audio folder; a phone takes the offer.
        let out_dir = project_dir.join("audio");
        let slug: String = clip
            .name
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == ' ' {
                    c
                } else {
                    '-'
                }
            })
            .take(60)
            .collect();
        let suggested = format!("{}.wav", slug.trim());
        // The dialog on the next turn of the event loop, never from here:
        // this is reached from a right-click menu's row, which reports
        // through a timer, and a blocking dialog opened inside a timer's
        // callback spins the loop back into Slint's timers - "Recursion in
        // timer code", and the app aborts (on macOS; Windows and Linux
        // were spared only by their dialogs not pumping the loop). And
        // with the studio unborrowed, for the timers that borrow it; see
        // `host::after_dialog`.
        let name = clip.name.clone();
        let title = t("lib.exportAudio");
        let family = t("lib.wavAudio");
        let ask_in = out_dir.clone();
        crate::host::after_dialog(
            move || {
                if cfg!(any(target_os = "android", target_os = "ios")) {
                    return Some(ask_in.join(&suggested));
                }
                let chosen = crate::platform::save_file(
                    &title,
                    &ask_in,
                    &suggested,
                    (family.as_str(), &["wav"]),
                )?;
                Some(
                    if chosen
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
                    {
                        chosen
                    } else {
                        chosen.with_extension("wav")
                    },
                )
            },
            move |studio, _, _, file| {
                studio.write_clip_audio(&name, pieces, duration, file, out_dir);
            },
        );
    }

    /// Mixes `pieces` - one clip's sound, `duration` seconds of it - into
    /// `file`, and puts the file in the bin, selected. The second half of
    /// [`Studio::export_clip_audio`], after the person has said where.
    fn write_clip_audio(
        &mut self,
        name: &str,
        pieces: Vec<concat_media::audio::AudioClip>,
        duration: f64,
        file: std::path::PathBuf,
        out_dir: std::path::PathBuf,
    ) {
        let out_dir = file
            .parent()
            .map(std::path::Path::to_path_buf)
            .unwrap_or(out_dir);
        let written = file.to_string_lossy().into_owned();
        log::info!(
            "export audio: {name} ({duration:.2}s, {} piece(s)) to {written}",
            pieces.len(),
        );
        self.notify(&t("studio.exportingAudio"), false);
        spawn_in_project(
            move || -> Result<concat_host::media::MediaSummary, String> {
                std::fs::create_dir_all(&out_dir)
                    .map_err(|error| format!("could not create {}: {error}", out_dir.display()))?;
                concat_media::audio::mix_to_file(&pieces, duration, &file)
                    .map_err(|error| error.to_string())?;
                concat_host::media::probe(&file.to_string_lossy())
            },
            move |studio, _, _, result| match result {
                Ok(summary) => {
                    let item = summary.to_new_media();
                    let path = item.path.clone();
                    let minted = studio.apply(Command::AddMedia { item }).or_else(|| {
                        studio
                            .project()
                            .media
                            .iter()
                            .find(|item| item.path == path)
                            .map(|item| item.id.clone())
                    });
                    // In view and in hand: the bin on all its media, with
                    // the new file the one thing selected.
                    studio.media.filter = MediaFilter::All;
                    studio.media.selected.clear();
                    if let Some(id) = minted {
                        studio.media.selected.insert(id);
                    }
                    studio.media_jump += 1;
                    studio.notify(&tf("studio.audioExported", &[&written]), false);
                }
                Err(error) => {
                    log::warn!("export audio: {error}");
                    studio.notify(&tf("studio.couldNotExportAudio", &[&error]), true);
                }
            },
        );
    }

    /// The Enhance switch in the inspector, for the one selected clip. A
    /// clip with a copy shows it or its original; one without starts
    /// Enhance on the switch going on, and stops a run on it going off.
    pub fn clip_enhance_set(&mut self, on: bool) {
        let Some(id) = self.sole_selection() else {
            return;
        };
        let Some(clip) = self.clip(&id) else {
            return;
        };
        if self.locked(&clip.track_id) {
            return;
        }
        if clip.enhanced.is_some() {
            self.apply(Command::SetClipEnhanced { clip_id: id, on });
            self.request_preview();
        } else if on {
            self.enhance_clip(&id);
        } else if self.enhance_jobs.contains_key(&id) {
            self.host.enhancers.cancel();
        }
    }

    pub fn enhance_clip(&mut self, id: &str) {
        if !self.enhance_jobs.is_empty() || self.host.enhancers.is_busy() {
            self.notify(&t("studio.enhanceAlreadyWorkOne"), true);
            return;
        }
        let Some(clip) = self.clip(id).cloned() else {
            return;
        };
        if clip.kind != model::ClipKind::Video && clip.kind != model::ClipKind::Image {
            return;
        }
        let Some(media) = self
            .project()
            .media
            .iter()
            .find(|item| item.id == clip.media_id)
            .cloned()
        else {
            return;
        };
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let project = std::path::PathBuf::from(session.path());
        let (width, height) = (media.width.unwrap_or(0), media.height.unwrap_or(0));
        if width == 0 || height == 0 {
            self.notify(&t("studio.fileHasNoPicture"), true);
            return;
        }
        let factor = concat_vision::enhance::factor_for(width, height);
        let still = clip.kind == model::ClipKind::Image;
        let Some(target) = concat_host::enhance::target_for(&project, &media.path, factor, still)
        else {
            self.notify(&tf("studio.couldNotRead", &[&media.path]), true);
            return;
        };
        let request = EnhanceRequest {
            media_path: media.path.clone(),
            still,
            factor,
            target,
        };
        let clip_id = clip.id.clone();
        self.enhance_jobs.insert(clip_id.clone(), (false, 0.0));
        self.notify(
            &tf("studio.enhancingWayFrameFrame", &[&media.name, &factor]),
            false,
        );
        let enhancers = Arc::clone(&self.host.enhancers);
        let epoch = crate::host::project_epoch();
        spawn_in_project(
            move || {
                let mut last = (false, -1.0f32);
                let reporting = clip_id.clone();
                let result = enhancers.enhance(&request, &mut |progress| {
                    let now = match progress {
                        concat_host::enhance::Progress::Fetching { received, total } => {
                            (true, received as f32 / total.max(1) as f32)
                        }
                        concat_host::enhance::Progress::Analysing(fraction) => (false, fraction),
                    };
                    if now.0 != last.0 || now.1 - last.1 >= 0.01 {
                        last = now;
                        let id = reporting.clone();
                        on_ui_in_project(epoch, move |studio, _, _| {
                            if let Some(held) = studio.enhance_jobs.get_mut(&id) {
                                *held = now;
                            }
                        });
                    }
                });
                (clip_id, result)
            },
            |studio, _, _, (clip_id, result)| {
                studio.enhance_jobs.remove(&clip_id);
                match result {
                    Ok(path) => studio.adopt_enhanced(&clip_id, &path),
                    Err(error) if error.contains("cancelled") => {}
                    Err(error) => studio.notify(&tf("studio.enhance", &[&error]), true),
                }
            },
        );
    }

    /// Writes the span clip `id` covers of its file backwards into the
    /// project's cache, and points the clip at the copy when it is there;
    /// see `concat_host::reverse`. The import is never touched.
    pub fn reverse_clip(&mut self, id: &str) {
        if !self.reverse_jobs.is_empty() || self.host.reversers.is_busy() {
            self.notify(&t("studio.reverseAlreadyWorkOne"), true);
            return;
        }
        let Some(clip) = self.clip(id).cloned() else {
            return;
        };
        let audio_only = match clip.kind {
            model::ClipKind::Video => false,
            model::ClipKind::Audio => true,
            _ => return,
        };
        if self.locked(&clip.track_id) {
            return;
        }
        let Some(media) = self
            .project()
            .media
            .iter()
            .find(|item| item.id == clip.media_id)
            .cloned()
        else {
            return;
        };
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let project = std::path::PathBuf::from(session.path());
        // The span the clip covers of the file: its in-point, and its
        // length at its rate - the curve's mean when it has one, which is
        // what `speed` holds then.
        let start = clip.source_start;
        let covered = clip.duration * clip.speed;
        let Some(target) =
            concat_host::reverse::target_for(&project, &media.path, start, covered, audio_only)
        else {
            self.notify(&tf("studio.couldNotRead", &[&media.path]), true);
            return;
        };
        let request = concat_host::ReverseRequest {
            media_path: media.path.clone(),
            audio_only,
            start,
            duration: covered,
            target,
        };
        let clip_id = clip.id.clone();
        self.reverse_jobs.insert(clip_id.clone(), 0.0);
        self.notify(&tf("studio.reversing", &[&media.name]), false);
        let reversers = Arc::clone(&self.host.reversers);
        let epoch = crate::host::project_epoch();
        spawn_in_project(
            move || {
                let mut last = -1.0f32;
                let reporting = clip_id.clone();
                let result = reversers.reverse(&request, &mut |fraction| {
                    if fraction - last >= 0.01 {
                        last = fraction;
                        let id = reporting.clone();
                        on_ui_in_project(epoch, move |studio, _, _| {
                            if let Some(held) = studio.reverse_jobs.get_mut(&id) {
                                *held = fraction;
                            }
                        });
                    }
                });
                (clip_id, start, covered, result)
            },
            |studio, _, _, (clip_id, start, covered, result)| {
                studio.reverse_jobs.remove(&clip_id);
                match result {
                    Ok(path) => studio.adopt_reversed(&clip_id, &path, start, covered),
                    Err(error) if error.contains("cancelled") => {}
                    Err(error) => studio.notify(&tf("studio.reverse", &[&error]), true),
                }
            },
        );
    }

    /// Points clip `id` at the reversed copy at `path`, its in-point at
    /// the copy's own zero, and shows it - unless the clip has moved on
    /// its file meanwhile, in which case the copy covers a span the clip
    /// no longer shows, and the ask has to be made again.
    fn adopt_reversed(&mut self, id: &str, path: &std::path::Path, start: f64, covered: f64) {
        let Some(clip) = self.clip(id).cloned() else {
            return;
        };
        if (clip.source_start - start).abs() > 1e-6
            || (clip.duration * clip.speed - covered).abs() > 1e-6
        {
            self.notify(&t("studio.clipChangedWhileBeing"), true);
            return;
        }
        let summary = match media::probe(&path.to_string_lossy()) {
            Ok(summary) => summary,
            Err(error) => {
                self.notify(&tf("studio.reverse", &[&error]), true);
                return;
            }
        };
        let mut item = summary.to_new_media();
        if let Some(original) = self.project().media.iter().find(|m| m.id == clip.media_id) {
            item.name = format!("{} (reversed)", original.name);
        }
        item.origin = Some(model::MediaOrigin::Processed);
        self.apply(Command::ReplaceClipMedia {
            clip_id: id.to_owned(),
            item,
            source_start: Some(0.0),
            enhanced: false,
        });
        self.request_preview();
        self.notify(&t("studio.reversedClipNowShows"), false);
    }

    /// Points clip `id` at the enhanced copy at `path`, probed the way an
    /// import is, and shows it.
    fn adopt_enhanced(&mut self, id: &str, path: &std::path::Path) {
        let summary = match media::probe(&path.to_string_lossy()) {
            Ok(summary) => summary,
            Err(error) => {
                self.notify(&tf("studio.enhance", &[&error]), true);
                return;
            }
        };
        let mut item = summary.to_new_media();
        // The bin names the copy after the original, marked, rather than
        // after its cache file.
        if let Some(original) = self
            .clip(id)
            .and_then(|clip| self.project().media.iter().find(|m| m.id == clip.media_id))
        {
            item.name = format!("{} (enhanced)", original.name);
        }
        self.apply(Command::ReplaceClipMedia {
            clip_id: id.to_owned(),
            item,
            source_start: None,
            enhanced: true,
        });
        self.request_preview();
        self.notify(&t("studio.enhancedClipNowShows"), false);
    }

    /// The Mode row: 0 off, 1 automatic, 2 custom. The chroma rows are the
    /// inspector's own business, on the chain.
    pub fn cutout_mode(&mut self, mode: i32) {
        let Some(id) = self.sole_selection() else {
            return;
        };
        let Some(clip) = self.clip(&id).cloned() else {
            return;
        };
        let held = clip.cutout.clone().unwrap_or_else(model::Cutout::auto);
        let cutout = match mode {
            1 => Some(model::Cutout {
                mode: model::CutoutMode::Auto,
                ..held
            }),
            2 => Some(model::Cutout {
                mode: model::CutoutMode::Custom,
                ..held
            }),
            _ => None,
        };
        if mode == 2 {
            self.painting = true;
        }
        if cutout != clip.cutout {
            self.apply(Command::SetClipCutout {
                clip_id: id,
                cutout,
            });
        }
    }

    /// The name an analysis runs under: the media and what it keeps.
    fn analysis_key(media_id: &str, subject: model::Subject) -> String {
        format!("{media_id}:{}", subject.key())
    }

    /// The Subject row: 0 automatic, 1 person, 2 object. A change means
    /// other masks, which the analysis notices on its own.
    pub fn cutout_subject(&mut self, index: i32) {
        let Some(id) = self.sole_selection() else {
            return;
        };
        let Some(cutout) = self.clip(&id).and_then(|clip| clip.cutout.clone()) else {
            return;
        };
        let subject = match index {
            1 => model::Subject::Person,
            2 => model::Subject::Object,
            _ => model::Subject::Auto,
        };
        if cutout.subject == subject {
            return;
        }
        self.apply(Command::SetClipCutout {
            clip_id: id,
            cutout: Some(model::Cutout { subject, ..cutout }),
        });
    }

    /// Takes every stroke off the selected clip's cutout, keeping it custom.
    pub fn cutout_clear(&mut self) {
        let Some(id) = self.sole_selection() else {
            return;
        };
        let Some(cutout) = self.clip(&id).and_then(|clip| clip.cutout.clone()) else {
            return;
        };
        if cutout.strokes.is_empty() {
            return;
        }
        self.apply(Command::SetClipCutout {
            clip_id: id,
            cutout: Some(model::Cutout {
                strokes: Vec::new(),
                ..cutout
            }),
        });
    }

    pub fn cutout_tool(&mut self, index: i32) {
        self.brush = index.clamp(0, BRUSHES.len() as i32 - 1) as usize;
    }

    pub fn cutout_size(&mut self, size: f32) {
        self.brush_size = f64::from(size).clamp(model::MIN_BRUSH, model::MAX_BRUSH);
    }

    pub fn cutout_painting(&mut self, on: bool) {
        self.painting = on;
        if !on {
            self.pending_stroke = None;
        }
        // The monitor's view changes with the brushes: tinted while they
        // are out, cut when they are put away.
        self.request_preview();
    }

    /// The picture a press on the stage would paint: the one selected clip,
    /// under the playhead, with a custom cutout, while painting is on.
    fn paint_target(&self) -> Option<Clip> {
        if !self.painting {
            return None;
        }
        let id = self.sole_selection()?;
        let clip = self.clip(&id)?;
        let custom = clip
            .cutout
            .as_ref()
            .is_some_and(|cutout| cutout.mode == model::CutoutMode::Custom);
        if !custom || !clip.kind.is_visual() || self.locked(&clip.track_id) {
            return None;
        }
        self.stage_clips()
            .into_iter()
            .any(|shown| shown.id == clip.id)
            .then(|| clip.clone())
    }

    /// Where a stage point lands in the source picture, in the fractions a
    /// stroke is stored in: the footprint's turn undone, then the crop and
    /// the flips, the same way a decoded pixel finds its mask.
    fn stage_to_source(&self, clip: &Clip, x: f64, y: f64) -> [f64; 2] {
        let footprint = self.footprint(clip);
        let (width, height) = self.output_size();
        let (width, height) = (f64::from(width.max(1)), f64::from(height.max(1)));
        let dx = (x - footprint.cx) * width;
        let dy = (y - footprint.cy) * height;
        let (sin, cos) = footprint.rotation.to_radians().sin_cos();
        let along = dx * cos + dy * sin;
        let down = -dx * sin + dy * cos;
        let px = along / (footprint.w * width).max(1e-6) + 0.5;
        let py = down / (footprint.h * height).max(1e-6) + 0.5;
        let mapping = concat_vision::Mapping {
            crop: clip
                .crop
                .map(|crop| {
                    [
                        crop.left as f32,
                        crop.top as f32,
                        crop.right as f32,
                        crop.bottom as f32,
                    ]
                })
                .unwrap_or([0.0; 4]),
            flip_h: clip.flip_h,
            flip_v: clip.flip_v,
        };
        let (u, v) = mapping.source_of(px as f32, py as f32);
        [f64::from(u), f64::from(v)]
    }

    /// The stroke in flight as the stage draws it: path commands over a
    /// 1000 × 1000 viewbox, the line's width as a fraction of the stage,
    /// and whether it is taking away. Empty between strokes.
    fn stroke_overlay(&self) -> (String, f32, bool) {
        if let Some(drawing) = self.draw_overlay() {
            return drawing;
        }
        let Gesture::Paint {
            clip,
            tool,
            size,
            screen,
            ..
        } = &self.gesture
        else {
            return self
                .pending_stroke
                .clone()
                .unwrap_or((String::new(), 0.0, false));
        };
        let Some((first, rest)) = screen.split_first() else {
            return (String::new(), 0.0, false);
        };
        let mut path = format!("M {:.1} {:.1}", first.0 * 1000.0, first.1 * 1000.0);
        // A single press still draws a dot: a line to where it already is,
        // with round caps.
        for (x, y) in rest.iter().chain(rest.is_empty().then_some(first)) {
            path.push_str(&format!(" L {:.1} {:.1}", x * 1000.0, y * 1000.0));
        }
        // The brush is `size` of the picture's width; on the stage that is
        // `size` of the picture's footprint.
        let width = self
            .clip(clip)
            .map(|clip| self.footprint(clip).w)
            .unwrap_or(1.0) as f32
            * *size as f32;
        let erase = matches!(
            tool,
            model::BrushTool::Eraser | model::BrushTool::SmartEraser
        );
        (path, width, erase)
    }

    // ── edits from menus and the tray ──

    /// Split at `at`: every selected clip the instant runs through, or every
    /// clip at all when nothing is selected.
    pub fn split_at(&mut self, at: f32, only_selected: bool) {
        let selection = self.selection.clone();
        let at = f64::from(at);
        let victims: Vec<String> = self
            .timeline()
            .clips
            .iter()
            .filter(|clip| {
                clip.start + f64::from(MIN_DURATION) < at
                    && at < clip.start + clip.duration - f64::from(MIN_DURATION)
                    && (!only_selected || selection.is_empty() || selection.contains(&clip.id))
                    && !self.locked(&clip.track_id)
            })
            .map(|clip| clip.id.clone())
            .collect();
        if !victims.is_empty() {
            self.apply(Command::SplitClips {
                clip_ids: victims,
                time: at,
            });
        }
    }

    /// Save Video Frame As: the frame under the playhead, drawn at the
    /// output's full size rather than the monitor's, written as a PNG
    /// where the person says - offered in the project's frames folder -
    /// and put in the bin, selected, so it can go straight onto a lane as
    /// a freeze frame. The dialog opens on the next turn of the loop, as
    /// every dialog reached from a right-click menu must.
    /// https://github.com/jub0t/Concat/issues/238
    pub fn save_frame_as(&mut self) {
        if self.playing {
            self.pause();
        }
        let Some((clips, settings)) = self.preview_clips() else {
            return;
        };
        let Some(project_dir) = self
            .session
            .as_ref()
            .map(|session| std::path::PathBuf::from(session.path()))
        else {
            return;
        };
        let (width, height) = self.output_size();
        let spec = concat_host::preview::FrameSpec {
            time: f64::from(self.playhead),
            width,
            height,
            moving: false,
            proxy: false,
            color_space: self.project().active().video.color_space,
        };
        let monitor = self.host.monitor.clone();
        let out_dir = project_dir.join("frames");
        let suggested = format!(
            "frame-{}.png",
            frames_timecode(self.playhead, self.frame_rate()).replace(':', ".")
        );
        let title = t("lib.saveFrame");
        let family = t("lib.pngImage");
        // With the studio unborrowed too; see `host::after_dialog`.
        crate::host::after_dialog(
            move || {
                if cfg!(any(target_os = "android", target_os = "ios")) {
                    return Some(out_dir.join(&suggested));
                }
                let chosen = crate::platform::save_file(
                    &title,
                    &out_dir,
                    &suggested,
                    (family.as_str(), &["png"]),
                )?;
                Some(
                    if chosen
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
                    {
                        chosen
                    } else {
                        chosen.with_extension("png")
                    },
                )
            },
            move |studio, _, _, file| {
                let written = file.to_string_lossy().into_owned();
                studio.notify(&t("studio.savingFrame"), false);
                spawn_in_project(
                    move || -> Result<concat_host::media::MediaSummary, String> {
                        let pixels = monitor.frame(clips, &settings, spec)?;
                        let image = image::RgbaImage::from_raw(width, height, pixels)
                            .ok_or_else(|| "the frame came back the wrong size".to_owned())?;
                        if let Some(parent) = file.parent() {
                            std::fs::create_dir_all(parent).map_err(|error| {
                                format!("could not create {}: {error}", parent.display())
                            })?;
                        }
                        image.save(&file).map_err(|error| error.to_string())?;
                        concat_host::media::probe(&file.to_string_lossy())
                    },
                    move |studio, _, _, result| match result {
                        Ok(summary) => {
                            let item = summary.to_new_media();
                            let path = item.path.clone();
                            let minted = studio.apply(Command::AddMedia { item }).or_else(|| {
                                studio
                                    .project()
                                    .media
                                    .iter()
                                    .find(|item| item.path == path)
                                    .map(|item| item.id.clone())
                            });
                            studio.media.filter = MediaFilter::All;
                            studio.media.selected.clear();
                            if let Some(id) = minted {
                                studio.media.selected.insert(id);
                            }
                            studio.media_jump += 1;
                            studio.notify(&tf("studio.frameSaved", &[&written]), false);
                        }
                        Err(error) => {
                            log::warn!("save frame: {error}");
                            studio.notify(&tf("studio.couldNotSaveFrame", &[&error]), true);
                        }
                    },
                );
            },
        );
    }

    /// The volume line on clip `id` dragged to `gain`: written the way the
    /// inspector's knob writes it - the constant, or a key at the playhead
    /// once the volume rides - and committed as the knob's moves are, one
    /// undo step a drag. The clip need not be selected.
    pub fn clip_volume(&mut self, id: &str, gain: f32) {
        let Some(clip) = self.clip(id) else {
            return;
        };
        if self.locked(&clip.track_id) {
            return;
        }
        let at = place_in(clip, self.playhead);
        // An edit still waiting to land on other clips - words typed into a
        // title a moment ago - lands first; otherwise this drag took over
        // its echo and its targets, and the words went with them (audit
        // 2026-10-04, #9). The drag's own moves share one commit.
        if self.commit_pending && self.commit_targets != [id] {
            self.flush_commit();
        }
        self.commit_targets = vec![id.to_owned()];
        self.begin_echo();
        if let Some(clip) = self.echo_clip_mut(id) {
            write_keyable(
                clip,
                model::KeyProperty::Volume,
                f64::from(gain).clamp(0.0, 2.0),
                at,
            );
        }
        self.clip_commit();
    }

    /// The split tool pressed clip `id` at `seconds`: one cut, on the
    /// frame nearest the press, unless it is within a frame of either end
    /// or the lane is locked.
    /// https://github.com/jub0t/Concat/issues/241
    pub fn split_clip_at(&mut self, id: &str, seconds: f32) {
        let Some(clip) = self.clip(id).cloned() else {
            return;
        };
        if self.locked(&clip.track_id) {
            return;
        }
        let fps = f64::from(self.frame_rate().max(1.0));
        let at = (f64::from(seconds) * fps).round() / fps;
        let edge = f64::from(MIN_DURATION);
        if at <= clip.start + edge || at >= clip.start + clip.duration - edge {
            return;
        }
        self.flush_commit();
        self.apply(Command::SplitClips {
            clip_ids: vec![id.to_owned()],
            time: at,
        });
    }

    /// Delete: the selection goes, and the hole stays unless the timeline
    /// is magnetic.
    pub fn delete_selected(&mut self) {
        // A selected transition goes on its own, leaving a plain cut.
        if self.selected_transition().is_some() {
            self.remove_transition();
            return;
        }
        let magnetic = self.prefs.magnetic;
        self.remove_selected(magnetic);
    }

    /// Ripple delete (⇧⌫): the selection goes and each lane closes behind
    /// it, so a rough cut needs no dragging-left after every deletion.
    /// https://github.com/jub0t/Concat/issues/106
    pub fn ripple_delete_selected(&mut self) {
        if self.selected_transition().is_some() {
            self.remove_transition();
            return;
        }
        self.remove_selected(true);
    }

    fn remove_selected(&mut self, ripple: bool) {
        let doomed: Vec<String> = self
            .selection
            .iter()
            .filter(|id| {
                self.clip(id)
                    .is_some_and(|clip| !self.locked(&clip.track_id))
            })
            .cloned()
            .collect();
        if !doomed.is_empty() {
            self.apply(Command::RemoveClips {
                clip_ids: doomed,
                ripple,
            });
        }
        self.selection.clear();
    }

    pub fn merge_blocked(&self) -> Option<String> {
        if self.selection.len() != 2 {
            return Some(t("studio.selectTwoClipsMerge"));
        }
        why_not_merge(self.timeline(), &self.selection)
    }

    pub fn merge(&mut self) {
        if self.merge_blocked().is_some() {
            return;
        }
        let ids = self.selection.clone();
        let kept = ids[0].clone();
        self.apply(Command::MergeClips { clip_ids: ids });
        self.selection = if self.clip(&kept).is_some() {
            vec![kept]
        } else {
            Vec::new()
        };
    }

    /// A freeze frame at the playhead on a picture clip.
    ///
    /// Video extracts a JPEG still into the project cache; images reuse their
    /// media. The engine command splits the clip, inserts the hold, and
    /// ripples the rest of the lane.
    pub fn freeze_at_playhead(&mut self) {
        self.freeze_for(1.0);
    }

    /// Freezes the picture under the playhead for `duration` seconds and
    /// selects the still; see `freeze_at_playhead`. The still's id, when
    /// one was made.
    pub fn freeze_for(&mut self, duration: f64) -> Option<String> {
        let at = f64::from(self.playhead);
        let edge = f64::from(MIN_DURATION);
        let target = self
            .menu_target
            .clone()
            .or_else(|| self.sole_selection())
            .and_then(|id| self.clip(&id).cloned())
            .filter(|clip| {
                (clip.kind == model::ClipKind::Video || clip.kind == model::ClipKind::Image)
                    && !self.locked(&clip.track_id)
                    && at > clip.start + edge
                    && at < clip.start + clip.duration - edge
            });
        let Some(clip) = target else {
            self.notify(&t("draw.parkToFreeze"), true);
            return None;
        };
        let source_clip = clip.clone();

        let still = if clip.kind == model::ClipKind::Video {
            let Some(media) = self
                .project()
                .media
                .iter()
                .find(|item| item.id == clip.media_id)
            else {
                return None;
            };
            let Some(session) = self.session.as_ref() else {
                return None;
            };
            let project_path = session.path().to_owned();
            let source_time = clip.source_start + (at - clip.start) * clip.speed;
            let frame = match media::still_at(&media.path, source_time, 1280) {
                Ok(frame) => frame,
                Err(error) => {
                    self.notify(&error, true);
                    return None;
                }
            };
            let bytes = match jpeg(&frame, 4) {
                Ok(bytes) => bytes,
                Err(error) => {
                    self.notify(&format!("{error}"), true);
                    return None;
                }
            };
            let key = format!(
                "freeze-{}-{}.jpg",
                clip.id,
                (source_time * 1000.0).round() as i64
            );
            if let Err(error) = media::write_artwork(&project_path, &key, &bytes) {
                self.notify(&error, true);
                return None;
            }
            let path = format!("{project_path}/cache/{key}");
            match media::probe(&path) {
                Ok(summary) => Some(summary.to_new_media()),
                Err(error) => {
                    self.notify(&error, true);
                    return None;
                }
            }
        } else {
            None
        };

        let created = self.apply(Command::FreezeFrame {
            clip_id: clip.id,
            time: at,
            duration: Some(duration),
            still,
        });
        if let Some(id) = created.as_ref() {
            self.selection = vec![id.clone()];
            self.carry_players_onto(&source_clip, at, id);
        }
        created
    }

    /// The players the football analysis found in `source` at timeline
    /// instant `at`, written as the analysis of the still `still` was
    /// frozen from it: one instant, which a still holds at every time, so
    /// a drawing or a mark on the frozen frame can still keep to a player.
    fn carry_players_onto(&self, source: &Clip, at: f64, still: &str) {
        let Some(from) = self.football_request(&source.id) else {
            return;
        };
        let Some(set) = concat_vision::football::shared(&from.dir()) else {
            return;
        };
        let Some(to) = self.football_request(still) else {
            return;
        };
        if to.media_path == from.media_path {
            return;
        }
        let seconds = source.source_start + (at - source.start) * source.speed;
        let frozen = concat_vision::football::TrackSet {
            version: set.version,
            model: set.model.clone(),
            aspect: set.aspect,
            samples: vec![concat_vision::football::Sample {
                ms: 0,
                boxes: set.at(seconds),
            }],
            team_colours: set.team_colours.clone(),
        };
        if let Err(error) = frozen.save(&to.dir()) {
            log::warn!("football: could not carry the players onto the still: {error}");
            return;
        }
        let roster = concat_vision::football::Roster::load(&from.dir());
        if !roster.players.is_empty()
            && let Err(error) = roster.save(&to.dir())
        {
            log::warn!("football: could not carry the names onto the still: {error}");
        }
        // And the pitch's calibration at that instant, so the board works
        // on the still: four points of the picture and where the source's
        // map put them.
        use concat_vision::football::pitch;
        let calibration = pitch::Calibration::load(&from.dir());
        let motion = pitch::Motion::of(&set);
        let ms = (seconds.max(0.0) * 1000.0).round() as u64;
        if let Some(map) = pitch::map_at(&calibration, &motion, ms) {
            let pairs: Vec<[f64; 4]> = [(0.25, 0.4), (0.75, 0.4), (0.8, 0.85), (0.2, 0.85)]
                .into_iter()
                .filter_map(|(u, v)| pitch::apply(&map, (u, v)).map(|(x, y)| [u, v, x, y]))
                .collect();
            if pairs.len() == 4 {
                let carried = pitch::Calibration {
                    anchors: vec![pitch::Anchor { ms: 0, pairs }],
                };
                if let Err(error) = carried.save(&to.dir()) {
                    log::warn!("football: could not carry the calibration onto the still: {error}");
                }
            }
        }
    }

    /// A copy of `source` laid after it. Three commands, because a clip's
    /// in-point and length are set by trims, not by placement.
    /// Duplicate every unlocked selected clip (right-to-left by start so
    /// neighbours do not stack). Falls back to the menu target when the
    /// selection is empty.
    pub fn duplicate_selected(&mut self) {
        let mut sources: Vec<Clip> = self
            .selection
            .iter()
            .filter_map(|id| self.clip(id).cloned())
            .filter(|clip| !self.locked(&clip.track_id))
            .collect();
        #[allow(clippy::collapsible_if)]
        if sources.is_empty() {
            if let Some(id) = self.menu_target.clone() {
                if let Some(clip) = self.clip(&id).cloned() {
                    if !self.locked(&clip.track_id) {
                        sources.push(clip);
                    }
                }
            }
        }
        if sources.is_empty() {
            return;
        }
        sources.sort_by(|left, right| right.start.total_cmp(&left.start));
        let mut last = None;
        for source in &sources {
            self.duplicate(source);
            last = self.selection.first().cloned();
        }
        if let Some(id) = last {
            self.selection = vec![id];
        }
    }

    /// A copy of `source` laid right after it on its own lane, selected.
    pub fn duplicate(&mut self, source: &Clip) {
        let end = source.start + source.duration;
        let track_id = source.track_id.clone();
        if let Some(created) = self.place_copy(source, &track_id, end, None) {
            self.selection = vec![created];
        }
    }

    /// Copy: the selected clips, or `anchor` alone when it is not among
    /// them - the clip a menu was opened on. Each is held with the row of
    /// its lane, so a paste into another timeline lands it on the same row
    /// there; the media is the project's, shared by every timeline. The
    /// held clips are kept in order of their start, the earliest first.
    pub fn copy_clips(&mut self, anchor: Option<&str>) {
        let mut ids: Vec<String> = self.selection.clone();
        if let Some(anchor) = anchor
            && !ids.iter().any(|id| id == anchor)
        {
            ids = vec![anchor.to_owned()];
        }
        let tracks = &self.timeline().tracks;
        let mut held: Vec<Held> = ids
            .iter()
            .filter_map(|id| {
                let clip = self.clip(id)?.clone();
                let row = tracks.iter().position(|track| track.id == clip.track_id)?;
                Some(Held { row, clip })
            })
            .collect();
        if held.is_empty() {
            return;
        }
        held.sort_by(|left, right| {
            left.clip
                .start
                .total_cmp(&right.clip.start)
                .then(left.row.cmp(&right.row))
        });
        self.clipboard = held;
    }

    /// Paste: the held clips as a group, its shape kept, as one undo step.
    /// After `after` and on its lane - the clip a menu was opened on, or
    /// the one selected - with the rest of the group on the rows around it
    /// as they were copied; or, with nothing to follow, at the playhead on
    /// the rows they were copied from. In another timeline the rows are
    /// matched by number, which is what a paste between timelines wanted
    /// and never got: the held lane's id was the other timeline's, and the
    /// add was refused without a word (#290). The new clips are selected.
    pub fn paste_clips(&mut self, after: Option<&str>) {
        if self.clipboard.is_empty() {
            return;
        }
        let held = self.clipboard.clone();
        let first_row = held[0].row;
        let tracks: Vec<String> = self
            .timeline()
            .tracks
            .iter()
            .map(|track| track.id.clone())
            .collect();
        let (base, shift) = match after.and_then(|id| self.clip(id).cloned()) {
            Some(anchor) => {
                let row = tracks
                    .iter()
                    .position(|id| *id == anchor.track_id)
                    .unwrap_or(first_row);
                let shift =
                    isize::try_from(row).unwrap_or(0) - isize::try_from(first_row).unwrap_or(0);
                (anchor.start + anchor.duration, shift)
            }
            None => (f64::from(self.playhead), 0),
        };
        let plan = paste_plan(&held, base, shift, tracks.len(), &|row| {
            tracks.get(row).is_some_and(|id| self.locked(id))
        });
        self.flush_commit();
        let mut created = Vec::new();
        for (held, place) in held.iter().zip(plan) {
            let Some((row, start)) = place else {
                continue;
            };
            if let Some(id) = self.place_copy(&held.clip, &tracks[row], start, Some("paste")) {
                created.push(id);
            }
        }
        // One step for the lot, and the next paste a step of its own.
        if let Some(session) = self.session.as_mut() {
            session.end_gesture();
        }
        if !created.is_empty() {
            self.selection = created;
        }
    }

    /// Lays a copy of `source` on `track_id` starting at `start` - its
    /// trims, mix, placement, speed, keys and effects with it - and says
    /// the new clip's id. `gesture` folds the commands into one undo step
    /// with whatever else names it; `None` leaves them steps of their own.
    ///
    /// Footage is added at normal speed, trimmed to the footage the source
    /// covers, then retimed: the speed command keeps the footage covered
    /// and recomputes the length, so the copy ends up the source's length
    /// over the source's frames.
    fn place_copy(
        &mut self,
        source: &Clip,
        track_id: &str,
        start: f64,
        gesture: Option<&str>,
    ) -> Option<String> {
        if source.kind == model::ClipKind::Text {
            return self.apply_in(
                gesture,
                Command::AddTextClip {
                    above: false,
                    track_id: Some(track_id.to_owned()),
                    start,
                    style: source.text.clone(),
                    duration: Some(source.duration),
                    offset_y: Some(source.offset_y),
                },
            );
        }
        if source.kind == model::ClipKind::Shape {
            return self.apply_in(
                gesture,
                Command::AddShapeClip {
                    above: false,
                    track_id: Some(track_id.to_owned()),
                    start,
                    style: source.shape.clone(),
                    duration: Some(source.duration),
                    name: source.name.clone(),
                },
            );
        }
        let created = self.apply_in(
            gesture,
            Command::AddClip {
                media_id: source.media_id.clone(),
                track_id: track_id.to_owned(),
                start: (start - source.source_start).max(0.0),
                ripple: false,
            },
        )?;
        let placed = self.clip(&created).cloned()?;
        let mut commands = Vec::new();
        let head = start - placed.start;
        if head.abs() > 1e-6 {
            commands.push(Command::TrimClip {
                clip_id: created.clone(),
                edge: TrimEdge::Start,
                delta: head,
                ripple: false,
            });
        }
        // The footage the source covers, at normal speed.
        let covered = source.duration * source.speed;
        let after_head = placed.duration - head.max(0.0);
        let tail = covered - after_head;
        if tail.abs() > 1e-6 {
            commands.push(Command::TrimClip {
                clip_id: created.clone(),
                edge: TrimEdge::End,
                delta: tail,
                ripple: false,
            });
        }
        if (source.speed - 1.0).abs() > 1e-9 {
            commands.push(Command::SetClipSpeed {
                clip_id: created.clone(),
                speed: source.speed,
            });
        }
        if source.speed_curve.is_some() {
            commands.push(Command::SetClipSpeedCurve {
                clip_id: created.clone(),
                curve: source.speed_curve.clone(),
            });
        }
        commands.push(Command::SetClipTransform {
            clip_id: created.clone(),
            scale: Some(source.scale),
            offset_x: Some(source.offset_x),
            offset_y: Some(source.offset_y),
            rotation: Some(source.rotation),
            stretch_x: Some(source.stretch_x),
            stretch_y: Some(source.stretch_y),
        });
        for key in &source.keys {
            commands.push(Command::SetClipKey {
                clip_id: created.clone(),
                property: key.property,
                at: key.at,
                value: key.value,
                ease: key.ease,
            });
        }
        commands.push(Command::UpdateClip {
            clip_id: created.clone(),
            patch: ClipPatch {
                name: Some(format!("{} copy", source.name)),
                volume: Some(source.volume),
                fade_in: Some(source.fade_in),
                fade_out: Some(source.fade_out),
                opacity: Some(source.opacity),
                preserve_pitch: Some(source.preserve_pitch),
                muted: source.muted,
                flip_h: Some(source.flip_h),
                flip_v: Some(source.flip_v),
                blend: Some(source.blend.clone()),
                crop: Some(source.crop),
                filters: Some(source.filters.clone()),
                video_effects: Some(source.video_effects.clone()),
                ..ClipPatch::default()
            },
        });
        self.apply_in(gesture, Command::Batch { commands });
        Some(created)
    }

    /// What the tray's sound and word tools may do to the selection: one
    /// clip with sound for Captions to start on, one title for Speak to
    /// read. Hints, not gates - both sheets open without them.
    fn sound_tools(&self) -> (bool, bool) {
        let Some(clip) = self.sole_selection().and_then(|id| self.clip(&id)) else {
            return (false, false);
        };
        (
            self.clip_has_sound(clip),
            clip.kind == model::ClipKind::Text,
        )
    }

    /// What the selection offers the tray's picture verbs: a picture on an
    /// unlocked lane, to mirror or turn; with one clip selected, a video or
    /// a still the playhead is inside of, to freeze there; and one video
    /// or sound clip to reverse, while no reverse is running.
    fn transform_tools(&self) -> (bool, bool, bool) {
        let at = f64::from(self.playhead);
        let edge = f64::from(MIN_DURATION);
        let sole = self.selection.len() == 1;
        let mut picture = false;
        let mut freezable = false;
        let mut reversible = false;
        for clip in self.selection.iter().filter_map(|id| self.clip(id)) {
            if self.locked(&clip.track_id) {
                continue;
            }
            let video = clip.kind == model::ClipKind::Video;
            if sole
                && self.reverse_jobs.is_empty()
                && (video || clip.kind == model::ClipKind::Audio)
            {
                reversible = true;
            }
            if clip.kind == model::ClipKind::Audio {
                continue;
            }
            picture = true;
            if sole
                && (video || clip.kind == model::ClipKind::Image)
                && at > clip.start + edge
                && at < clip.start + clip.duration - edge
            {
                freezable = true;
            }
        }
        (picture, freezable, reversible)
    }

    /// Where a clip's sound is: whether it is a video clip whose sound is
    /// still on its picture, to take off, and whether it is a picture whose
    /// sound is off it - or that sound itself - to put back.
    fn sound_placement(&self, clip: &Clip) -> (bool, bool) {
        let detached = self
            .timeline()
            .clips
            .iter()
            .any(|other| other.detached_from.as_deref() == Some(clip.id.as_str()));
        let video = clip.kind == model::ClipKind::Video;
        (
            video && !detached,
            (video && detached)
                || (clip.kind == model::ClipKind::Audio && clip.detached_from.is_some()),
        )
    }

    /// Whether the clip has sound to transcribe: an audio clip, or a video
    /// clip whose file carries an audio stream. A silent video is not a
    /// sound source, however much it looks like one.
    pub(crate) fn clip_has_sound(&self, clip: &Clip) -> bool {
        match clip.kind {
            model::ClipKind::Audio => true,
            model::ClipKind::Video => self
                .project()
                .media_by_id(&clip.media_id)
                .is_some_and(|media| media.has_audio),
            _ => false,
        }
    }

    // ── projects ──

    /// Opens a project as the session and leaves the launch screen, or
    /// says why it could not.
    pub fn open_project(&mut self, info: ProjectInfo) -> Result<(), String> {
        log::info!("project: opening {}", info.path);
        if self
            .host
            .open_projects
            .claim(&info.path, concat_api::Holder::Window)
            .is_err()
        {
            return Err(t("studio.projectOpenThroughRemote"));
        }
        match Session::open_info(&info) {
            Ok(session) => {
                if let Err(error) = projects::remember(&self.host.dirs.config, &info) {
                    log::warn!("{error}");
                }
                self.pause();
                // The proxies of media no longer in the project, or read
                // as a range they no longer are, go now: nothing else
                // sweeps cache/proxy.
                let project_dir = std::path::Path::new(session.path());
                let kept: HashSet<std::path::PathBuf> = session
                    .project()
                    .media
                    .iter()
                    .filter_map(|item| {
                        concat_host::proxy::path_for(
                            project_dir,
                            &item.path,
                            item.color_range.map(concat_export::engine_range),
                            item.color_space.is_hdr(),
                        )
                    })
                    .collect();
                let swept = concat_host::proxy::sweep(project_dir, &kept);
                if swept > 0 {
                    log::info!("{swept} stale proxies swept from {}", project_dir.display());
                }
                self.session = Some(session);
                crate::host::next_project_epoch();
                self.forget_art();
                self.art_failed.clear();
                self.art_stamp = None;
                self.proxied.clear();
                self.echo = None;
                self.dirty = false;
                self.project_name = info.name.clone();
                self.export.name = projects::folder_name(&info.name);
                self.selection.clear();
                self.transition_selected = None;
                self.media.selected.clear();
                self.playhead = 0.0;
                self.handle(crate::panes::Msg::Timeline(
                    crate::panes::timeline::TimelineMsg::Reset,
                ));
                self.on_start = false;
                self.handle(crate::panes::Msg::Monitor(
                    crate::panes::monitor::MonitorMsg::Opened,
                ));
                self.recents = projects::list(&self.host.dirs.config);
                self.host.monitor.clear();
                self.audition = None;
                self.revision += 1;
                self.flat = None;
                self.sync_audio();
                self.request_media_art();
                self.request_preview();
                self.ensure_cutouts();
                self.ensure_regions();

                // Effects this build no longer has, named once as the project
                // opens: their links stay in the clips, marked not installed.
                let retired = self
                    .session
                    .as_ref()
                    .map(|session| retired_effects(session.project()))
                    .unwrap_or_default();
                if !retired.is_empty() {
                    self.notify(&tf("studio.retiredEffects", &[&retired.join(", ")]), false);
                }
                // Effects that open as the ones standing in for them: the
                // document changes when it is next saved, so say so.
                let upgraded = self
                    .session
                    .as_ref()
                    .map(|session| upgraded_effects(session.upgraded()))
                    .unwrap_or_default();
                if !upgraded.is_empty() {
                    self.notify(
                        &tf("studio.upgradedEffects", &[&upgraded.join(", ")]),
                        false,
                    );
                }
                // Clips that played backwards or moved by a preset in an
                // earlier version, which this one does not read: named,
                // with the way back where there is one.
                let retired = self
                    .session
                    .as_ref()
                    .map(|session| session.retired().clone())
                    .unwrap_or_default();
                if !retired.reversed.is_empty() {
                    self.notify(
                        &tf("studio.clipsNowForwards", &[&few(&retired.reversed)]),
                        false,
                    );
                }
                if !retired.animated.is_empty() {
                    self.notify(
                        &tf("studio.clipsLostAnimations", &[&few(&retired.animated)]),
                        false,
                    );
                }

                // Log missing media to file for debugging
                if let Some(session) = &self.session {
                    let missing = session.project().missing_media();
                    if !missing.is_empty() {
                        let log_path = std::path::Path::new(&info.path)
                            .join("cache")
                            .join("missing_media.log");
                        if let Ok(mut file) = std::fs::File::create(&log_path) {
                            use std::io::Write;
                            let _ = writeln!(file, "{} media files missing:", missing.len());
                            for m in &missing {
                                let _ = writeln!(file, "  - {} ({})", m.name, m.path);
                            }
                        }

                        self.handle(crate::panes::Msg::Relink(
                            crate::panes::relink::RelinkMsg::Show(missing),
                        ));
                    }
                }
                Ok(())
            }
            Err(error) => {
                self.host.open_projects.release(&info.path);
                Err(error)
            }
        }
    }

    /// Clears all cached artwork and waveforms from the current project.
    pub fn clear_project_cache(&mut self) {
        let Some(path) = self
            .session
            .as_ref()
            .map(|session| session.path().to_string())
        else {
            return;
        };
        match concat_host::projects::clear_cache(&path) {
            Ok(count) => self.notify(&format!("Cleared {count} cache files"), false),
            Err(error) => self.notify(&format!("Cache clear failed: {error}"), true),
        }
        self.request_media_art();
        self.request_preview();
    }

    /// Saves, then closes the session and returns to the launch screen.
    pub fn close_project(&mut self) {
        log::info!("project: closing {}", self.project_name);
        // Words still being typed land in the save, not on the floor.
        self.flush_commit();
        self.pause();
        self.host.cutouts.cancel();
        self.host.enhancers.cancel();
        self.host.reversers.cancel();
        self.enhance_jobs.clear();
        self.reverse_jobs.clear();
        self.cutout_jobs.clear();
        self.region_job = None;
        if let Some(session) = self.session.as_mut() {
            let (path, document) = session.prepare_save(None);
            if let Err(error) = projects::save(&path, &document) {
                self.notify(&tf("studio.couldNotSave", &[&error]), true);
                return;
            }
        }
        self.autosave.stop();
        if let Some(session) = self.session.as_ref() {
            self.host.open_projects.release(session.path());
        }
        self.session = None;
        // Whatever a worker still brings back for this project is dropped
        // at delivery; the sheets that were waiting on one stop waiting.
        crate::host::next_project_epoch();
        self.forget_art();
        self.art_failed.clear();
        self.art_stamp = None;
        self.proxied.clear();
        self.captions.running = false;
        self.captions.progress = 0.0;
        self.speech.running = false;
        self.speech.progress = 0.0;
        self.source = None;
        // A take running as the project closes stops, keeping its file;
        // the next project's speakers are not left silent.
        self.voiceover = Default::default();
        self.host.playback.set_muted(false);
        self.echo = None;
        self.dirty = false;
        self.selection.clear();
        self.transition_selected = None;
        self.gesture = Gesture::None;
        self.handle(crate::panes::Msg::Monitor(
            crate::panes::monitor::MonitorMsg::Closed,
        ));
        self.host.monitor.clear();
        self.audition = None;
        self.revision += 1;
        self.flat = None;
        self.host
            .playback
            .set_clips(std::path::PathBuf::new(), Vec::new());
        self.on_start = true;
        self.recents = projects::list(&self.host.dirs.config);
    }

    /// Posters for the recents that have none yet, decoded on a worker.
    fn request_posters(&mut self) {
        let wanted: Vec<String> = self
            .recents
            .iter()
            .map(|project| project.path.clone())
            .filter(|path| !self.posters.contains_key(path) && !self.posters_pending.contains(path))
            .collect();
        for path in wanted {
            self.posters_pending.insert(path.clone());
            spawn(
                move || {
                    let cached = media::poster_cache(&path);
                    // A project with no poster - nothing on it, or nothing
                    // but black - keeps the film mark; see `poster_frame`.
                    let made = media::poster_frame(&path).is_ok();
                    (path, made.then_some(cached))
                },
                |studio, _, _, (path, cached)| {
                    studio.posters_pending.remove(&path);
                    if let Some(poster) = cached.and_then(|cached| image_at(&cached)) {
                        studio.posters.insert(path, poster);
                    }
                },
            );
        }
    }

    /// Routes one message to its pane and applies it. The pane is taken out
    /// of the studio for the duration, so its `update` can be handed the
    /// rest of the window without borrowing itself twice; a pane never
    /// reads its own slot on the studio.
    pub fn handle(&mut self, msg: crate::panes::Msg) {
        // The activity trail: what the person did, for the log. A burst
        // - a wheel's zooms - is one line and a count; see `Burst`.
        if let Some(activity) = crate::panes::activity(&msg) {
            self.note_activity(activity);
        }
        match msg {
            crate::panes::Msg::Export(msg) => {
                let mut pane = std::mem::take(&mut self.export);
                pane.update(msg, self);
                self.export = pane;
            }
            crate::panes::Msg::Settings(msg) => {
                let mut pane = std::mem::take(&mut self.settings);
                pane.update(msg, self);
                self.settings = pane;
            }
            crate::panes::Msg::Captions(msg) => {
                let mut pane = std::mem::take(&mut self.captions);
                pane.update(msg, self);
                self.captions = pane;
            }
            crate::panes::Msg::Speech(msg) => {
                let mut pane = std::mem::take(&mut self.speech);
                pane.update(msg, self);
                self.speech = pane;
            }
            crate::panes::Msg::Voiceover(msg) => {
                let mut pane = std::mem::take(&mut self.voiceover);
                pane.update(msg, self);
                self.voiceover = pane;
            }
            crate::panes::Msg::Relink(msg) => {
                let mut pane = std::mem::take(&mut self.relink);
                pane.update(msg, self);
                self.relink = pane;
            }
            crate::panes::Msg::Project(msg) => {
                let mut pane = std::mem::take(&mut self.project_sheet);
                pane.update(msg, self);
                self.project_sheet = pane;
            }
            crate::panes::Msg::Start(msg) => {
                let mut pane = std::mem::take(&mut self.start);
                pane.update(msg, self);
                self.start = pane;
            }
            crate::panes::Msg::Media(msg) => {
                let mut pane = std::mem::take(&mut self.media);
                pane.update(msg, self);
                self.media = pane;
            }
            crate::panes::Msg::Monitor(msg) => {
                let mut pane = std::mem::take(&mut self.monitor);
                pane.update(msg, self);
                self.monitor = pane;
            }
            crate::panes::Msg::Timeline(msg) => {
                let mut pane = std::mem::take(&mut self.lanes);
                pane.update(msg, self);
                self.lanes = pane;
            }
        }
    }

    // ── speech ──

    /// Packs the open project into the template library.
    /// Where the user's own looks live: one package folder each.
    pub fn looks_dir(dirs: &concat_host::dirs::AppDirs) -> std::path::PathBuf {
        dirs.config.join("effects")
    }

    /// Where the effect cards are kept: drawn once, read on every launch.
    pub fn cards_dir(dirs: &concat_host::dirs::AppDirs) -> std::path::PathBuf {
        dirs.data.join("cards")
    }

    /// Plans every picture package's card against the catalogue as it
    /// stands, sweeps away the cards no package draws any more, and draws
    /// the missing ones on a thread of their own, on a compositor sharing
    /// the monitor's device - or, while a round is drawing already, asks
    /// for another when it ends.
    fn plan_cards(&mut self) {
        let dir = Self::cards_dir(&self.host.dirs);
        let cards = concat_host::cards::cards(Catalogue::builtin(), &dir);
        concat_host::cards::prune(&dir, &cards);
        self.card_paths = cards
            .iter()
            .map(|card| (card.id.clone(), card.path.clone()))
            .collect();
        let missing: Vec<concat_host::cards::Card> =
            cards.into_iter().filter(|card| !card.is_drawn()).collect();
        if missing.is_empty() {
            return;
        }
        if self.cards_busy {
            self.cards_again = true;
            return;
        }
        self.cards_busy = true;
        let compositor = self.host.monitor.sibling();
        let spawned = std::thread::Builder::new()
            .name("concat cards".to_owned())
            .spawn(move || {
                concat_host::cards::draw_all(&dir, &missing, compositor, |ids, done| {
                    crate::host::on_ui(move |studio, _, _| studio.cards_arrived(ids, done));
                });
            });
        if let Err(error) = spawned {
            log::warn!("cards: {error}");
            self.cards_busy = false;
        }
    }

    /// Cards have landed: `ids` are read again the next time the shelves
    /// are built. With `done` the round is over, and another is planned
    /// when the catalogue changed while it drew.
    fn cards_arrived(&mut self, ids: Vec<String>, done: bool) {
        {
            let mut art = self.look_art.borrow_mut();
            for id in &ids {
                art.remove(id);
            }
        }
        self.cards_drawn += ids.len() as u64;
        if done {
            self.cards_busy = false;
            if std::mem::take(&mut self.cards_again) {
                self.plan_cards();
            }
        }
    }

    /// Rebuilds the catalogue from the built-ins and the packages in
    /// `looks_dir`, and says in the window what would not load: the first
    /// reason and how many more, with every reason in the log, so an
    /// author sees a manifest or shader fault where they are looking
    /// rather than in a file they may not know exists. Returns how many
    /// failed. With `announce`, a clean reload is reported too, with the
    /// folder, which is how a newcomer learns where packages go.
    pub fn reload_packages(&mut self, announce: bool) -> usize {
        /// A custom package's shader runs once over a picture this many
        /// pixels a side before the package is offered: enough that a loop
        /// bounded in the thousands per pixel shows, and a real frame's
        /// worth of work does not.
        const TRIAL_SIDE: u32 = 512;
        /// How long that trial may take before the package is refused.
        const TRIAL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);
        let dir = Self::looks_dir(&self.host.dirs);
        // Looks an earlier build imported, onto the format 2 look first.
        for id in concat_effects::looks::upgrade(&dir) {
            log::info!("{id}: the imported look now reads its table in light");
        }
        // The trials run on a compositor of their own on the window's
        // device, never the monitor's: one that times out is dead from
        // then on, and a refused package must not take the preview with
        // it (audit 2026-10-04, #6). A dead one is replaced for the next.
        let monitor = &self.host.monitor;
        let mut trialist = None;
        let errors = Catalogue::install_with(&dir, &mut |package| {
            for pass in package.trial_passes() {
                if trialist.is_none() {
                    trialist = monitor.sibling();
                }
                let Some(gpu) = trialist.as_mut() else {
                    return Ok(());
                };
                let tried = gpu.trial_at(&pass, TRIAL_SIDE, TRIAL_TIMEOUT);
                if gpu.is_dead() {
                    trialist = None;
                }
                if let Err(why) = tried {
                    return Err(format!("its shader failed its trial: {why}"));
                }
            }
            Ok(())
        });
        for error in &errors {
            log::warn!("package: {error}");
        }
        // Every picture package's card: those drawn before are read as the
        // shelves need them, the rest drawn on a thread of their own.
        self.plan_cards();
        // A changed shader means a changed card: the stills are read again.
        self.look_art.borrow_mut().clear();
        // The folder as the watch will next see it, cards included, so a
        // reload does not read as a change and set off another.
        self.packages_seen = concat_effects::package_stamp(&dir);
        self.packages_pending = None;
        if let Some(first) = errors.first() {
            let message = if errors.len() == 1 {
                tf("studio.customPackageDidNot", &[first])
            } else {
                tf("studio.customPackagesDidNot", &[&errors.len(), first])
            };
            self.notify(&message, true);
        } else if announce {
            let count = Catalogue::builtin()
                .packages()
                .filter(|package| package.folder.is_some())
                .count();
            self.notify(
                &tf("studio.loadedCustomPackagesFrom", &[&count, &dir.display()]),
                false,
            );
        }
        errors.len()
    }

    /// Starts the watch on the effects folder: a poll every two seconds,
    /// and a reload once two polls in a row see the same changed folder.
    /// A package written or dropped in while the app runs then shows up on
    /// its own, and one that will not load says why, without a restart or
    /// the button.
    pub fn watch_packages(&mut self) {
        self.packages_watch.start(
            slint::TimerMode::Repeated,
            std::time::Duration::from_secs(2),
            || {
                crate::host::Shell::with(|shell, app| {
                    let changed = shell.studio.borrow_mut().poll_packages();
                    if changed {
                        shell.studio.borrow_mut().refresh_art();
                        shell.studio.borrow().publish(&app, &shell.models);
                    }
                });
            },
        );
    }

    /// One tick of the watch: whether the catalogue was rebuilt.
    fn poll_packages(&mut self) -> bool {
        let now = concat_effects::package_stamp(&Self::looks_dir(&self.host.dirs));
        if now == self.packages_seen {
            self.packages_pending = None;
            return false;
        }
        if self.packages_pending != Some(now) {
            // Seen once; a copy may still be in progress. Next tick decides.
            self.packages_pending = Some(now);
            return false;
        }
        self.reload_packages(true);
        true
    }

    /// Imports one or more `.cube` tables as looks: each becomes a package
    /// folder under `looks_dir` - a manifest that names the table, a shader
    /// that reads it, the table (see `concat_effects::looks`) - and the
    /// catalogue is rebuilt so the Filters page shows them, their cards
    /// drawn by their own shaders.
    pub fn import_lut(&mut self) {
        let Some(paths) = crate::platform::pick_files(
            &t("studio.importLut"),
            Some((t("studio.lut").as_str(), &["cube"])),
        ) else {
            return;
        };
        let dir = Self::looks_dir(&self.host.dirs);
        let mut imported = 0;
        for path in paths {
            match concat_effects::looks::import(&dir, &path) {
                Ok(id) => {
                    self.look_art.borrow_mut().remove(&id);
                    imported += 1;
                }
                Err(error) => {
                    self.notify(
                        &tf("studio.couldNotImport", &[&path.display(), &error]),
                        true,
                    );
                }
            }
        }
        if imported == 0 {
            return;
        }
        // A look that will not load has its reason on screen; the count of
        // the rest would only cover it.
        if self.reload_packages(false) > 0 {
            return;
        }
        self.library[0].query.clear();
        self.notify(&tf("studio.importedLooksFindThem", &[&imported]), false);
    }

    pub fn save_template(&mut self) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let document = session.document();
        let settings = session.settings();
        let path = session.path().to_owned();
        let name = format!("{} template", self.project_name);
        let config = self.host.dirs.config.clone();
        spawn(
            move || templates::save(&config, &document, &settings, &path, &name),
            |studio, _, _, result| match result {
                Ok(info) => studio.notify(&tf("studio.savedTemplate", &[&info.name]), false),
                Err(error) => studio.notify(&error, true),
            },
        );
    }

    // ── notices ──

    /// Raises the bottom-right notice. Every caller is some other handler
    /// that has just done - or refused to do - the thing it is reporting, so
    /// this only records; the publish that handler was going to make anyway
    /// is what puts it on screen.
    pub fn notify(&mut self, message: &str, failed: bool) {
        self.toast.token += 1;
        self.toast.message = message.into();
        self.toast.failed = failed;
    }

    // ── publishing ──

    pub fn publish(&self, app: &App, models: &Models) {
        self.publish_lanes(app, models);
        self.publish_chrome(app, models);
        self.publish_dock(app, models);
    }

    pub fn publish_dock(&self, _app: &App, models: &Models) {
        let out = self.dock_layout();
        sync(&models.seats, out.seats);
        sync(&models.dividers, out.dividers);
    }

    /// The window is a phone's: the lanes keep the playhead at their middle
    /// and glide under it. See `platform::phone` and `TimelinePane::centred`.
    pub fn set_phone(&mut self, phone: bool) {
        self.lanes.centred = phone;
        let playhead = self.playhead;
        self.lanes.recentre(playhead);
    }

    /// Shows the compact dock, or the wide one, keeping whichever is put
    /// away whole; see `COMPACT_WIDTH`.
    pub fn set_compact(&mut self, compact: bool) {
        if compact != self.compact {
            std::mem::swap(&mut self.dock, &mut self.dock_aside);
            self.compact = compact;
        }
    }

    pub fn dock_layout(&self) -> DockLayout {
        let mut out = DockLayout::default();
        let (width, height) = self.workspace;
        if width > SEAT_GAP * 2.0 && height > SEAT_GAP * 2.0 {
            lay_out(
                &self.dock,
                (
                    SEAT_GAP,
                    SEAT_GAP,
                    width - 2.0 * SEAT_GAP,
                    height - 2.0 * SEAT_GAP,
                ),
                &mut out,
            );
        }
        out
    }

    pub fn split_extent(&self, index: usize) -> Option<f32> {
        self.dock_layout().extents.get(index).copied()
    }

    pub fn split_ratio(&self, index: usize) -> Option<f32> {
        match self.dock.at(&self.dock.split_path(index)?) {
            Dock::Split { ratio, .. } => Some(*ratio),
            _ => None,
        }
    }

    /// The monitor's picture alone: what a frame arriving during playback
    /// changes.
    pub fn publish_frame(&self, app: &App) {
        app.global::<Editor>()
            .set_preview_frame(self.monitor.image.clone());
    }

    /// The timeline and the readouts that follow it: what runs on every
    /// event of a scrub, a drag, a trim or a knob.
    pub fn publish_lanes(&self, app: &App, models: &Models) {
        let editor = app.global::<Editor>();
        let project = self.project();
        sync(
            &models.tabs,
            project
                .timelines
                .iter()
                .map(|timeline| TimelineTabData {
                    id: timeline.id.as_str().into(),
                    name: timeline.name.as_str().into(),
                })
                .collect(),
        );

        let timeline = self.timeline();
        let mut top = 0.0;
        sync(
            &models.tracks,
            timeline
                .tracks
                .iter()
                .rev()
                .map(|lane| {
                    let height = self.lane_height(lane);
                    let row = TrackData {
                        id: lane.id.as_str().into(),
                        visible: lane.visible,
                        muted: lane.muted,
                        locked: self.locked(&lane.id),
                        size: self.lane_size(&lane.id),
                        height,
                        top,
                    };
                    top += height;
                    row
                })
                .collect(),
        );

        // Only the clips near the view: what the lanes show plus a screen
        // either side; see `TimelinePane::published_span`.
        let span = self.lanes.published_span();
        sync(
            &models.clips,
            timeline
                .clips
                .iter()
                .filter(|clip| {
                    crate::panes::timeline::TimelinePane::shows(
                        span,
                        clip.start as f32,
                        clip.duration as f32,
                    )
                })
                .map(|clip| {
                    let wave =
                        if matches!(clip.kind, model::ClipKind::Audio | model::ClipKind::Video) {
                            self.wave(clip)
                        } else {
                            Default::default()
                        };
                    ClipData {
                        id: clip.id.as_str().into(),
                        name: clip.name.as_str().into(),
                        kind: kind_of(clip),
                        row: self.row_of(&clip.track_id),
                        start: clip.start as f32,
                        duration: clip.duration as f32,
                        selected: self.selection.iter().any(|id| id == &clip.id),
                        transition_selected: self
                            .selected_transition()
                            .is_some_and(|picked| picked.id == clip.id),
                        fx: clip.video_effects.iter().any(|effect| effect.enabled),
                        transition_duration: clip
                            .transition_in
                            .as_ref()
                            .map(|transition| transition.duration as f32)
                            .unwrap_or(0.0),
                        fade_in: clip.fade_in as f32,
                        fade_out: clip.fade_out as f32,
                        volume: clip.volume as f32,
                        text_body: clip
                            .text
                            .as_ref()
                            .map(|text| text.content.as_str())
                            .unwrap_or_default()
                            .into(),
                        wave: wave.0,
                        wave_from: wave.1,
                        wave_span: wave.2,
                        strip: self.strip_of(clip),
                    }
                })
                .collect(),
        );

        // What is under the playhead. The topmost picture wins, the way the
        // compositor stacks them.
        let playhead = f64::from(self.playhead);
        let showing = timeline
            .clips
            .iter()
            .filter(|clip| {
                (clip.kind.is_visual()
                    || matches!(clip.kind, model::ClipKind::Text | model::ClipKind::Shape))
                    && clip.start <= playhead
                    && playhead < clip.start + clip.duration
            })
            .max_by_key(|clip| -self.row_of(&clip.track_id));
        editor.set_has_picture(showing.is_some());
        editor.set_preview_clip_name(
            showing
                .map(|clip| clip.name.as_str())
                .unwrap_or_default()
                .into(),
        );
        editor.set_preview_duration(self.duration());
        editor.set_playhead_free(!self.prefs.playhead_stops_at_end);
        editor.set_playing(self.playing);
        self.publish_frame(app);
        sync(&models.stage, self.stage_items());
        sync(&models.guides, self.stage_guides.clone());
        let (path, width, erase) = self.stroke_overlay();
        editor.set_stroke_path(path.into());
        editor.set_stroke_width(width);
        editor.set_stroke_erase(erase);
        editor.set_brush_tool(self.brush as i32);
        editor.set_brush_size(self.brush_size as f32);
        editor.set_painting(self.painting);

        editor.set_drop(match &self.drop {
            Some(plan) => DropData {
                active: true,
                kind: plan.kind,
                label: plan.label.as_str().into(),
                start: plan.start,
                duration: plan.duration,
                row: plan.row,
            },
            None => DropData::default(),
        });

        editor.set_selected_clip(self.selected());
        editor.set_selected_transition(self.selected_transition_data());
        // Written only when it differs: a fresh model is a change to every
        // binding that reads it, and this one is read on every publish.
        let labels = self.audio_track_labels();
        let shown = editor.get_audio_tracks();
        let same = shown.row_count() == labels.len()
            && labels
                .iter()
                .enumerate()
                .all(|(row, label)| shown.row_data(row).as_ref() == Some(label));
        if !same {
            editor.set_audio_tracks(slint::ModelRc::new(VecModel::from(labels)));
        }
        // The keyframe cluster's state, on its own global: every keyable row
        // in the inspector reads it, and none of them is threaded to.
        let keys = app.global::<Keyframes>();
        // The Keyframes panel: set whole, since a row holds a model of its
        // keys and cannot be compared for a diff.
        models.key_editor_rows.set_vec(self.key_editor_rows());
        app.global::<KeyEditor>().set_at(
            self.sole_selection()
                .and_then(|id| self.clip(&id))
                .map_or(0.0, |clip| place_in(clip, self.playhead) as f32),
        );
        let rows = self.key_rows();
        keys.set_available(!rows.is_empty());
        sync(&models.key_rows, rows);
        let fonts = Some(self.fonts_fingerprint());
        let searched = Some({
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            (fonts, &self.font_query).hash(&mut hasher);
            hasher.finish()
        });
        if self.fonts_published.get() != fonts || self.font_search_published.get() != searched {
            self.fonts_published.set(fonts);
            self.font_search_published.set(searched);
            let families = self.font_families();
            let query = self.font_query.trim().to_lowercase();
            sync(
                &models.font_matches,
                families
                    .iter()
                    .filter(|family| query.is_empty() || family.to_lowercase().contains(&query))
                    .map(SharedString::from)
                    .collect(),
            );
            sync(
                &models.font_families,
                families.into_iter().map(SharedString::from).collect(),
            );
        }
        editor.set_inspector_jump_token(self.inspector_jump.0);
        editor.set_library_audition(self.audition_of().unwrap_or("").into());
        editor.set_inspector_jump_tab(self.inspector_jump.1.into());
        editor.set_inspector_jump_section(self.inspector_jump.2.into());
        let active = project
            .timelines
            .iter()
            .position(|timeline| timeline.id == project.active_timeline_id)
            .unwrap_or(0);
        editor.set_timeline_current_tab(active as i32);
        editor.set_playhead(self.playhead);
        sync(&models.key_marks, self.key_marks());
        sync(&models.play_marks, self.montage_marks());
        self.publish_tagging(&editor, models);
        self.publish_analysis(&editor);
        self.publish_board(&editor, models);
        editor.set_mark_in(self.mark_in.unwrap_or(-1.0));
        editor.set_mark_out(self.mark_out.unwrap_or(-1.0));
        editor.set_scroll_left(self.lanes.scroll_left);
        editor.set_seconds_per_pixel(self.lanes.seconds_per_pixel);
        editor.set_duration_seconds(self.duration());
        editor.set_frame_rate(self.frame_rate());
        editor.set_tool(self.lanes.tool);
        editor.set_snap(self.lanes.snap);
        editor.set_magnetic(self.prefs.magnetic);
        editor.set_trim_follow(self.prefs.trim_follow);
        editor.set_preview_axis(self.prefs.preview_axis);
        editor.set_preview_axis_audio(self.prefs.preview_axis_audio);
        editor.set_pan_mode(self.lanes.pan_mode);
        editor.set_selected_count(self.selection.len() as i32);
        // The tray's undo and redo buttons grey out on these; the Edit menu
        // asks the session itself when it opens.
        // https://github.com/jub0t/Concat/issues/154
        let (can_undo, can_redo) = self.session.as_ref().map_or((false, false), |session| {
            (session.can_undo(), session.can_redo())
        });
        editor.set_can_undo(can_undo);
        editor.set_can_redo(can_redo);
        let (sound_selected, title_selected) = self.sound_tools();
        editor.set_sound_selected(sound_selected);
        editor.set_title_selected(title_selected);
        let (picture_selected, freezable, reversible) = self.transform_tools();
        editor.set_picture_selected(picture_selected);
        editor.set_freezable(freezable);
        editor.set_reversible(reversible);
        editor.set_merge_blocked_because(match self.merge_blocked() {
            Some(reason) => reason.into(),
            None => SharedString::new(),
        });

        // The selected clip's chains, as the inspector's two stacks.
        let (video, audio) = match self.sole_selection().and_then(|id| self.clip(&id)) {
            Some(clip) => (
                clip.video_effects
                    .iter()
                    .enumerate()
                    .map(|(index, effect)| EffectData {
                        id: index as i32 + 1,
                        name: label_of(&effect.id).into(),
                        audio: false,
                    })
                    .collect(),
                clip.filters
                    .iter()
                    .enumerate()
                    .map(|(index, filter)| EffectData {
                        id: index as i32 + 1000,
                        name: label_of(&filter.id).into(),
                        audio: true,
                    })
                    .collect(),
            ),
            None => (Vec::new(), Vec::new()),
        };
        sync(&models.video_effects, video);
        sync(&models.audio_effects, audio);

        // The same chains as the inspector's stacks see them: a row per
        // link and a row per knob, from the catalogue's manifests.
        let (visual, visual_params, sound, sound_params) =
            match self.sole_selection().and_then(|id| self.clip(&id)) {
                Some(clip) => {
                    let (visual, visual_params) = chain_rows(&clip.video_effects);
                    let (sound, sound_params) = chain_rows(&clip.filters);
                    (visual, visual_params, sound, sound_params)
                }
                None => (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            };
        sync(&models.applied_visual, visual);
        sync(&models.visual_params, visual_params);
        sync(
            &models.visual_colours,
            match self.sole_selection().and_then(|id| self.clip(&id)) {
                Some(clip) => chain_colours(&clip.video_effects),
                None => Vec::new(),
            },
        );
        sync(&models.applied_audio, sound);
        sync(&models.audio_params, sound_params);
        sync(
            &models.adjust_params,
            match self.sole_selection().and_then(|id| self.clip(&id)) {
                Some(clip) if clip.kind.is_visual() => {
                    adjust_rows(&clip.video_effects, self.key_point().map(|(_, at)| at))
                }
                _ => Vec::new(),
            },
        );

        // The wheels and curves: the chain's by link, the colour panel's
        // keyable at the playhead. Each curve's points go into the model
        // kept for it.
        let clip = self.sole_selection().and_then(|id| self.clip(&id));
        let (visual_wheels, visual_curves) =
            clip.map_or_else(Default::default, |clip| chain_grading(&clip.video_effects));
        let (adjust_wheels, adjust_curves) = match (clip, Catalogue::builtin().get(ADJUST_ID)) {
            (Some(clip), Some(package)) if clip.kind.is_visual() => grading_rows(
                package,
                clip.video_effects
                    .iter()
                    .find(|entry| entry.id == ADJUST_ID),
                -1,
                Some(self.key_point().map(|(_, at)| at)),
            ),
            _ => Default::default(),
        };
        let placed = |curves: Vec<(CurveData, Vec<CurvePointData>)>| -> Vec<CurveData> {
            let mut kept = models.curve_points.borrow_mut();
            curves
                .into_iter()
                .map(|(mut curve, points)| {
                    let model = Rc::clone(
                        kept.entry((curve.entry, curve.key.to_string()))
                            .or_insert_with(|| Rc::new(VecModel::default())),
                    );
                    sync(&model, points);
                    curve.points = ModelRc::from(model);
                    curve
                })
                .collect()
        };
        sync(&models.adjust_wheels, adjust_wheels);
        sync(&models.adjust_curves, placed(adjust_curves));
        // The chain's, one group a link, in chain order, each group's list
        // the model kept for its link.
        let visual_curves = placed(visual_curves);
        let mut entries: Vec<i32> = visual_wheels
            .iter()
            .map(|wheel| wheel.entry)
            .chain(visual_curves.iter().map(|curve| curve.entry))
            .collect();
        entries.sort_unstable();
        entries.dedup();
        let (mut wheel_groups, mut curve_groups) = (Vec::new(), Vec::new());
        for entry in entries {
            let wheels: Vec<WheelData> = visual_wheels
                .iter()
                .filter(|wheel| wheel.entry == entry)
                .cloned()
                .collect();
            if !wheels.is_empty() {
                let model = Rc::clone(
                    models
                        .link_wheels
                        .borrow_mut()
                        .entry(entry)
                        .or_insert_with(|| Rc::new(VecModel::default())),
                );
                sync(&model, wheels);
                wheel_groups.push(WheelGroupData {
                    entry,
                    wheels: ModelRc::from(model),
                });
            }
            let curves: Vec<CurveData> = visual_curves
                .iter()
                .filter(|curve| curve.entry == entry)
                .cloned()
                .collect();
            if !curves.is_empty() {
                let model = Rc::clone(
                    models
                        .link_curves
                        .borrow_mut()
                        .entry(entry)
                        .or_insert_with(|| Rc::new(VecModel::default())),
                );
                sync(&model, curves);
                curve_groups.push(CurveGroupData {
                    entry,
                    curves: ModelRc::from(model),
                });
            }
        }
        sync(&models.visual_wheels, wheel_groups);
        sync(&models.visual_curves, curve_groups);
    }

    // ── the effect libraries ──

    /// The view state of one library, or None for a shelf index the panel
    /// should never have sent.
    fn library_at(&mut self, shelf: i32) -> Option<&mut LibraryView> {
        self.library.get_mut(usize::try_from(shelf).ok()?)
    }

    /// The search box was typed in. A live query searches every shelf, so
    /// the strip is put away while one is running - see `shelves`.
    pub fn library_query(&mut self, shelf: i32, text: &str) {
        if let Some(view) = self.library_at(shelf) {
            view.query = text.to_owned();
        }
    }

    /// A shelf was picked. Typing and then picking a shelf means the shelf,
    /// so the query goes.
    pub fn library_group(&mut self, shelf: i32, index: i32) {
        if let Some(view) = self.library_at(shelf) {
            view.group = index.max(0);
            view.category.clear();
            view.query.clear();
            view.favourites = false;
        }
    }

    /// A category pill was picked.
    pub fn library_category(&mut self, shelf: i32, category: &str) {
        if let Some(view) = self.library_at(shelf) {
            view.category = category.to_owned();
            view.group = -1;
            view.query.clear();
        }
    }

    /// The star chip. Turning it on clears the query for the same reason
    /// picking a shelf does: two filters at once is a list nobody can
    /// predict the contents of.
    pub fn library_favourites(&mut self, shelf: i32, on: bool) {
        if let Some(view) = self.library_at(shelf) {
            view.favourites = on;
            view.query.clear();
        }
    }

    /// Star or unstar a package, and remember it. Not per library: a star is
    /// a fact about the package.
    pub fn library_favourite(&mut self, id: &str, on: bool) {
        let held = self.prefs.favourites.iter().any(|held| held == id);
        if on && !held {
            self.prefs.favourites.push(id.to_owned());
        } else if !on && held {
            self.prefs.favourites.retain(|held| held != id);
        } else {
            return;
        }
        self.prefs.save(&self.host.dirs);
    }

    // ── keyframes ──

    /// The keyable property a field names, or None for a field that is a
    /// constant and stays one.
    pub fn key_property_of(field: ClipField) -> Option<model::KeyProperty> {
        Some(match field {
            ClipField::Scale => model::KeyProperty::Scale,
            ClipField::OffsetX => model::KeyProperty::OffsetX,
            ClipField::OffsetY => model::KeyProperty::OffsetY,
            ClipField::Rotation => model::KeyProperty::Rotation,
            ClipField::Opacity => model::KeyProperty::Opacity,
            ClipField::Volume => model::KeyProperty::Volume,
            _ => return None,
        })
    }

    /// The selected clip and where the playhead is inside it, `0..=1`, or
    /// None when there is no sole selection or the playhead is outside it.
    ///
    /// Outside is not clamped to an end: a key put on from outside the clip
    /// would land on its first or last frame, which is never what the person
    /// pressing the diamond meant.
    pub fn key_point(&self) -> Option<(&Clip, f64)> {
        let clip = self.sole_selection().and_then(|id| self.clip(&id))?;
        if clip.duration <= 0.0 {
            return None;
        }
        let local = f64::from(self.playhead) - clip.start;
        (0.0..=clip.duration)
            .contains(&local)
            .then(|| (clip, local / clip.duration))
    }

    /// The keyframe cluster's six rows, in the order the `Keyframes` global
    /// indexes them. Empty when nothing can be keyed, which is what greys
    /// every cluster in the inspector at once.
    pub fn key_rows(&self) -> Vec<ClipKeyData> {
        let Some((clip, at)) = self.key_point() else {
            return Vec::new();
        };
        model::KeyProperty::ALL
            .iter()
            .map(|&property| {
                let (prev, next) = clip.keys_around(property, at);
                ClipKeyData {
                    field: key_field_of(property),
                    keyed: clip.is_keyed(property),
                    here: clip.key_at(property, at).is_some(),
                    prev: prev.is_some(),
                    next: next.is_some(),
                }
            })
            .collect()
    }

    /// The Keyframes pane's rows: the six properties of the selected clip,
    /// and every Adjust knob of it that carries keys, each with its keys
    /// along the clip - where, what value as a fraction of the knob's
    /// range, and the ease into it - what the knob is worth at the
    /// playhead, and the ride over the clip as a path for the curve view.
    pub fn key_editor_rows(&self) -> Vec<KeyRowData> {
        let Some(clip) = self.sole_selection().and_then(|id| self.clip(&id)) else {
            return Vec::new();
        };
        let at = place_in(clip, self.playhead);
        let near = |a: f64, b: f64| (a - b).abs() <= model::KEY_EPSILON;
        // A run of keys as marks: each with its value as a fraction of the
        // range and the ease into it, and the ride as a path - a hold to
        // the first key, a cubic per segment (the ease's control points are
        // exactly the curve's, in time and value), a hold after the last.
        let marks_and_curve = |keys: &[(f64, f64, model::KeyEase)],
                               to_fraction: &dyn Fn(f64) -> f64|
         -> (ModelRc<KeyMarkData>, String, i32) {
            let marks: Vec<KeyMarkData> = keys
                .iter()
                .enumerate()
                .map(|(index, (key_at, value, ease))| {
                    let [x1, y1, x2, y2] = ease.0;
                    KeyMarkData {
                        at: *key_at as f32,
                        here: near(*key_at, at),
                        value: to_fraction(*value).clamp(0.0, 1.0) as f32,
                        ease_x1: x1 as f32,
                        ease_y1: y1 as f32,
                        ease_x2: x2 as f32,
                        ease_y2: y2 as f32,
                        eased: index > 0,
                    }
                })
                .collect();
            let mut curve = String::new();
            if let (Some(first), Some(last)) = (marks.first(), marks.last()) {
                let y = |value: f32| 1.0 - value;
                curve.push_str(&format!(
                    "M 0 {:.4} L {:.4} {:.4}",
                    y(first.value),
                    first.at,
                    y(first.value)
                ));
                for pair in marks.windows(2) {
                    let (a, b) = (&pair[0], &pair[1]);
                    let (dx, dy) = (b.at - a.at, b.value - a.value);
                    curve.push_str(&format!(
                        " C {:.4} {:.4} {:.4} {:.4} {:.4} {:.4}",
                        a.at + b.ease_x1 * dx,
                        y(a.value + b.ease_y1 * dy),
                        a.at + b.ease_x2 * dx,
                        y(a.value + b.ease_y2 * dy),
                        b.at,
                        y(b.value)
                    ));
                }
                curve.push_str(&format!(" L 1 {:.4}", y(last.value)));
            }
            // The key the ease menu edits: the one at the playhead, else
            // the next, when it has a key behind it to ease from.
            let ease_index = marks
                .iter()
                .position(|mark| mark.eased && f64::from(mark.at) >= at - model::KEY_EPSILON)
                .map_or(-1, |index| index as i32);
            (ModelRc::new(VecModel::from(marks)), curve, ease_index)
        };
        let mut rows = Vec::new();
        for property in model::KeyProperty::ALL {
            // The same ranges and units as the inspector's knobs. The level
            // knob speaks decibels, floored where silence is.
            let (label, minimum, maximum, step, default_value, fmt, unit, unit_scale) =
                match property {
                    model::KeyProperty::Scale => (
                        t("common.scale"),
                        0.05,
                        8.0,
                        0.01,
                        1.0,
                        ParamFormat::Fraction,
                        "%",
                        100.0,
                    ),
                    model::KeyProperty::OffsetX => (
                        t("common.positionX"),
                        -1.0,
                        1.0,
                        0.005,
                        0.0,
                        ParamFormat::Centred,
                        "%",
                        100.0,
                    ),
                    model::KeyProperty::OffsetY => (
                        t("common.positionY"),
                        -1.0,
                        1.0,
                        0.005,
                        0.0,
                        ParamFormat::Centred,
                        "%",
                        100.0,
                    ),
                    model::KeyProperty::Rotation => (
                        t("common.rotation"),
                        -180.0,
                        180.0,
                        1.0,
                        0.0,
                        ParamFormat::Degrees,
                        "°",
                        1.0,
                    ),
                    model::KeyProperty::Opacity => (
                        t("common.opacity"),
                        0.0,
                        1.0,
                        0.01,
                        1.0,
                        ParamFormat::Fraction,
                        "%",
                        100.0,
                    ),
                    model::KeyProperty::Volume => (
                        t("common.level"),
                        -60.0,
                        24.0,
                        0.5,
                        0.0,
                        ParamFormat::Gain,
                        "",
                        1.0,
                    ),
                };
            let to_knob = |value: f64| knob_value(property, value);
            let keys: Vec<(f64, f64, model::KeyEase)> = clip
                .keys_on(property)
                .map(|key| (key.at, to_knob(key.value), key.ease))
                .collect();
            let (marks, curve, ease_index) =
                marks_and_curve(&keys, &|value| (value - minimum) / (maximum - minimum));
            let (prev, next) = clip.keys_around(property, at);
            rows.push(KeyRowData {
                field: key_field_of(property),
                param: SharedString::default(),
                label: label.into(),
                keys: marks,
                state: ClipKeyData {
                    field: key_field_of(property),
                    keyed: !keys.is_empty(),
                    here: clip.key_at(property, at).is_some(),
                    prev: prev.is_some(),
                    next: next.is_some(),
                },
                value: to_knob(shown(clip, property, at)) as f32,
                minimum: minimum as f32,
                maximum: maximum as f32,
                step,
                default_value,
                fmt,
                unit: unit.into(),
                unit_scale,
                curve: curve.into(),
                ease_index,
            });
        }
        if let Some(link) = clip
            .video_effects
            .iter()
            .find(|entry| entry.id == ADJUST_ID)
        {
            let inside = self.key_point().map(|(_, at)| at);
            for row in adjust_rows(&clip.video_effects, inside) {
                if !row.keyed {
                    continue;
                }
                let keys: Vec<(f64, f64, model::KeyEase)> = link
                    .keys_on(&row.key)
                    .iter()
                    .map(|key| (key.at, key.value, key.ease))
                    .collect();
                let (minimum, maximum) = (f64::from(row.min), f64::from(row.max));
                let (marks, curve, ease_index) = marks_and_curve(&keys, &|value| {
                    if maximum > minimum {
                        (value - minimum) / (maximum - minimum)
                    } else {
                        0.5
                    }
                });
                rows.push(KeyRowData {
                    field: ClipField::Scale,
                    param: row.key.clone(),
                    label: row.label.clone(),
                    keys: marks,
                    state: ClipKeyData {
                        field: ClipField::Scale,
                        keyed: true,
                        here: row.here,
                        prev: row.prev,
                        next: row.next,
                    },
                    value: row.value,
                    minimum: row.min,
                    maximum: row.max,
                    step: row.step,
                    default_value: row.default_value,
                    fmt: row.fmt,
                    unit: row.unit.clone(),
                    unit_scale: 1.0,
                    curve: curve.into(),
                    ease_index,
                });
            }
            // A wheel that rides is one row, drawn from its master's keys;
            // its puck's keys sit at the same instants and move with them.
            if let Some(package) = Catalogue::builtin().get(ADJUST_ID) {
                let (wheels, _) = grading_rows(package, Some(link), -1, Some(inside));
                for wheel in wheels.into_iter().filter(|wheel| wheel.keyed) {
                    let [_, _, master] = grading::wheel_keys(&wheel.key);
                    let keys: Vec<(f64, f64, model::KeyEase)> = link
                        .keys_on(&master)
                        .iter()
                        .map(|key| (key.at, key.value, key.ease))
                        .collect();
                    let (minimum, maximum) = (f64::from(wheel.min), f64::from(wheel.max));
                    let (marks, curve, ease_index) = marks_and_curve(&keys, &|value| {
                        if maximum > minimum {
                            (value - minimum) / (maximum - minimum)
                        } else {
                            0.5
                        }
                    });
                    rows.push(KeyRowData {
                        field: ClipField::Scale,
                        param: master.as_str().into(),
                        label: wheel.label.clone(),
                        keys: marks,
                        state: ClipKeyData {
                            field: ClipField::Scale,
                            keyed: true,
                            here: wheel.here,
                            prev: wheel.prev,
                            next: wheel.next,
                        },
                        value: wheel.master,
                        minimum: wheel.min,
                        maximum: wheel.max,
                        step: 0.01,
                        default_value: wheel.default_value,
                        fmt: format_of(""),
                        unit: SharedString::default(),
                        unit_scale: 1.0,
                        curve: curve.into(),
                        ease_index,
                    });
                }
            }
        }
        rows
    }

    /// Moves the key at `from` to `to` and gives it `fraction` of its
    /// row's range, keeping its ease: a drag of a handle in the curve view.
    pub fn drag_key(&mut self, field: ClipField, param: &str, from: f32, to: f32, fraction: f32) {
        let Some(clip) = self.sole_selection().and_then(|id| self.clip(&id)) else {
            return;
        };
        let clip_id = clip.id.clone();
        let (from, to) = (f64::from(from), f64::from(to).clamp(0.0, 1.0));
        let fraction = f64::from(fraction).clamp(0.0, 1.0);
        let commands = if param.is_empty() {
            let Some(property) = Self::key_property_of(field) else {
                return;
            };
            let Some(index) = clip.key_at(property, from) else {
                return;
            };
            let (minimum, maximum) = knob_range(property);
            let value = model_value(property, minimum + fraction * (maximum - minimum));
            let ease = clip.keys[index].ease;
            vec![
                Command::ClearClipKey {
                    clip_id: clip_id.clone(),
                    property,
                    at: from,
                },
                Command::SetClipKey {
                    clip_id,
                    property,
                    at: to,
                    value,
                    ease,
                },
            ]
        } else {
            let Some(entry) = clip
                .video_effects
                .iter()
                .position(|entry| entry.id == ADJUST_ID)
            else {
                return;
            };
            let link = &clip.video_effects[entry];
            let Some(index) = link.key_at(param, from) else {
                return;
            };
            let Some(row) = adjust_rows(&clip.video_effects, Some(from))
                .into_iter()
                .find(|row| row.key == param)
            else {
                return;
            };
            let (minimum, maximum) = (f64::from(row.min), f64::from(row.max));
            let value = minimum + fraction * (maximum - minimum);
            let ease = link.keys_on(param)[index].ease;
            let mut commands = vec![
                Command::ClearEffectKey {
                    clip_id: clip_id.clone(),
                    entry,
                    key: param.to_owned(),
                    at: from,
                },
                Command::SetEffectKey {
                    clip_id: clip_id.clone(),
                    entry,
                    key: param.to_owned(),
                    at: to,
                    value,
                    ease,
                },
            ];
            commands.extend(partners_moved(link, &clip_id, entry, param, from, to));
            commands
        };
        self.apply(Command::Batch { commands });
    }

    /// The selected clip's keys as instants on the timeline, every property
    /// and Adjust knob together, one diamond per instant: what the ruler
    /// draws while exactly one clip is selected (#188).
    pub fn key_marks(&self) -> Vec<f32> {
        let Some(clip) = self.sole_selection().and_then(|id| self.clip(&id)) else {
            return Vec::new();
        };
        let mut ats: Vec<f64> = model::KeyProperty::ALL
            .iter()
            .flat_map(|&property| clip.keys_on(property).map(|key| key.at))
            .collect();
        if let Some(link) = clip
            .video_effects
            .iter()
            .find(|entry| entry.id == ADJUST_ID)
        {
            ats.extend(link.keys.values().flatten().map(|key| key.at));
        }
        ats.sort_by(f64::total_cmp);
        ats.dedup_by(|a, b| (*a - *b).abs() <= model::KEY_EPSILON);
        ats.iter()
            .map(|at| (clip.start + at * clip.duration) as f32)
            .collect()
    }

    /// Moves every key of the selected clip that sits at the instant
    /// `from` to the instant `to`, as one undo step: what a dragged ruler
    /// diamond does.
    pub fn move_keys_at(&mut self, from: f32, to: f32) {
        let Some(clip) = self.sole_selection().and_then(|id| self.clip(&id)) else {
            return;
        };
        if clip.duration <= 0.0 {
            return;
        }
        let clip_id = clip.id.clone();
        let from_at = (f64::from(from) - clip.start) / clip.duration;
        let to_at = ((f64::from(to) - clip.start) / clip.duration).clamp(0.0, 1.0);
        let mut commands = Vec::new();
        for property in model::KeyProperty::ALL {
            let Some(index) = clip.key_at(property, from_at) else {
                continue;
            };
            let key = &clip.keys[index];
            commands.push(Command::ClearClipKey {
                clip_id: clip_id.clone(),
                property,
                at: key.at,
            });
            commands.push(Command::SetClipKey {
                clip_id: clip_id.clone(),
                property,
                at: to_at,
                value: key.value,
                ease: key.ease,
            });
        }
        if let Some(entry) = clip
            .video_effects
            .iter()
            .position(|entry| entry.id == ADJUST_ID)
        {
            let link = &clip.video_effects[entry];
            for name in link.keys.keys() {
                let Some(index) = link.key_at(name, from_at) else {
                    continue;
                };
                let key = &link.keys_on(name)[index];
                commands.push(Command::ClearEffectKey {
                    clip_id: clip_id.clone(),
                    entry,
                    key: name.clone(),
                    at: key.at,
                });
                commands.push(Command::SetEffectKey {
                    clip_id: clip_id.clone(),
                    entry,
                    key: name.clone(),
                    at: to_at,
                    value: key.value,
                    ease: key.ease,
                });
            }
        }
        if !commands.is_empty() {
            self.apply(Command::Batch { commands });
        }
    }

    /// Moves the playhead to `at` of the selected clip, `0..1`.
    pub fn jump_to_key(&mut self, at: f32) {
        let Some(clip) = self.sole_selection().and_then(|id| self.clip(&id)) else {
            return;
        };
        let (start, duration) = (clip.start, clip.duration);
        self.seek((start + f64::from(at).clamp(0.0, 1.0) * duration) as f32);
    }

    /// Moves the key at `from` to `to` on a property, or on the Adjust
    /// knob `param` names: the same value and ease at the new place, as
    /// one undo step.
    pub fn move_key(&mut self, field: ClipField, param: &str, from: f32, to: f32) {
        let Some(clip) = self.sole_selection().and_then(|id| self.clip(&id)) else {
            return;
        };
        let clip_id = clip.id.clone();
        let (from, to) = (f64::from(from), f64::from(to).clamp(0.0, 1.0));
        let commands = if param.is_empty() {
            let Some(property) = Self::key_property_of(field) else {
                return;
            };
            let Some(index) = clip.key_at(property, from) else {
                return;
            };
            let key = &clip.keys[index];
            let (value, ease) = (key.value, key.ease);
            vec![
                Command::ClearClipKey {
                    clip_id: clip_id.clone(),
                    property,
                    at: from,
                },
                Command::SetClipKey {
                    clip_id,
                    property,
                    at: to,
                    value,
                    ease,
                },
            ]
        } else {
            let Some(entry) = clip
                .video_effects
                .iter()
                .position(|entry| entry.id == ADJUST_ID)
            else {
                return;
            };
            let link = &clip.video_effects[entry];
            let Some(index) = link.key_at(param, from) else {
                return;
            };
            let key = &link.keys_on(param)[index];
            let (value, ease) = (key.value, key.ease);
            let mut commands = vec![
                Command::ClearEffectKey {
                    clip_id: clip_id.clone(),
                    entry,
                    key: param.to_owned(),
                    at: from,
                },
                Command::SetEffectKey {
                    clip_id: clip_id.clone(),
                    entry,
                    key: param.to_owned(),
                    at: to,
                    value,
                    ease,
                },
            ];
            commands.extend(partners_moved(link, &clip_id, entry, param, from, to));
            commands
        };
        self.apply(Command::Batch { commands });
    }

    /// Sets the ease into the key at `at`. A drag of the curve editor is
    /// many of these, folded into one undo step.
    pub fn set_key_ease(&mut self, field: ClipField, param: &str, at: f32, ease: [f32; 4]) {
        let Some(clip) = self.sole_selection().and_then(|id| self.clip(&id)) else {
            return;
        };
        let clip_id = clip.id.clone();
        let at = f64::from(at);
        let ease = model::KeyEase(ease.map(f64::from)).sane();
        let command = if param.is_empty() {
            let Some(property) = Self::key_property_of(field) else {
                return;
            };
            let Some(index) = clip.key_at(property, at) else {
                return;
            };
            Command::SetClipKey {
                clip_id,
                property,
                at,
                value: clip.keys[index].value,
                ease,
            }
        } else {
            let Some(entry) = clip
                .video_effects
                .iter()
                .position(|entry| entry.id == ADJUST_ID)
            else {
                return;
            };
            let link = &clip.video_effects[entry];
            if link.key_at(param, at).is_none() {
                return;
            }
            let mut commands: Vec<Command> = std::iter::once(param.to_owned())
                .chain(wheel_partners(param))
                .filter_map(|name| {
                    let index = link.key_at(&name, at)?;
                    Some(Command::SetEffectKey {
                        clip_id: clip_id.clone(),
                        entry,
                        value: link.keys_on(&name)[index].value,
                        key: name,
                        at,
                        ease,
                    })
                })
                .collect();
            if commands.len() == 1 {
                commands.remove(0)
            } else {
                Command::Batch { commands }
            }
        };
        self.apply_within("ease", command);
    }

    /// Moves the playhead to the nearest key behind (-1) or ahead (+1) on
    /// any knob of the selected clip.
    pub fn step_any_key(&mut self, delta: i32) {
        let Some((clip, at)) = self.key_point() else {
            return;
        };
        let mut candidates: Vec<f64> = model::KeyProperty::ALL
            .iter()
            .flat_map(|&property| clip.keys_on(property).map(|key| key.at))
            .collect();
        if let Some(link) = clip
            .video_effects
            .iter()
            .find(|entry| entry.id == ADJUST_ID)
        {
            candidates.extend(link.keys.values().flatten().map(|key| key.at));
        }
        let target = if delta < 0 {
            candidates
                .iter()
                .copied()
                .filter(|&key_at| key_at < at - model::KEY_EPSILON)
                .reduce(f64::max)
        } else {
            candidates
                .iter()
                .copied()
                .filter(|&key_at| key_at > at + model::KEY_EPSILON)
                .reduce(f64::min)
        };
        let Some(target) = target else {
            return;
        };
        let (start, duration) = (clip.start, clip.duration);
        self.seek((start + target * duration) as f32);
    }

    /// Takes every key off the selected clip: its properties and its
    /// Adjust knobs, as one undo step.
    pub fn clear_all_keys(&mut self) {
        let Some(clip) = self.sole_selection().and_then(|id| self.clip(&id)) else {
            return;
        };
        let clip_id = clip.id.clone();
        let mut commands: Vec<Command> = model::KeyProperty::ALL
            .iter()
            .filter(|&&property| clip.is_keyed(property))
            .map(|&property| Command::ClearClipKeys {
                clip_id: clip_id.clone(),
                property,
            })
            .collect();
        if let Some(entry) = clip
            .video_effects
            .iter()
            .position(|entry| entry.id == ADJUST_ID)
        {
            for key in clip.video_effects[entry].keys.keys() {
                commands.push(Command::ClearEffectKeys {
                    clip_id: clip_id.clone(),
                    entry,
                    key: key.clone(),
                });
            }
        }
        if !commands.is_empty() {
            self.apply(Command::Batch { commands });
        }
    }

    /// Puts a key on the field at the playhead, or takes off the one there.
    ///
    /// The value a new key gets is whatever the property is worth at that
    /// instant - the constant on a clip with no keys, and the ride's own
    /// value on one that has them. That is what makes the diamond safe to
    /// press: it never moves the picture, it only says "hold this here".
    pub fn toggle_key(&mut self, field: ClipField) {
        let Some(property) = Self::key_property_of(field) else {
            return;
        };
        let Some((clip, at)) = self.key_point() else {
            return;
        };
        let clip_id = clip.id.clone();
        let command = if clip.key_at(property, at).is_some() {
            Command::ClearClipKey {
                clip_id,
                property,
                at,
            }
        } else {
            let value = clip.value_at(property, at);
            // The ease of whichever key this one is joining behind, so
            // laying a run of keys down does not alternate between shapes.
            // The first key on a property has nothing to inherit and gets
            // the straight line.
            let ease = clip
                .keys_on(property)
                .rfind(|key| key.at < at)
                .map_or(model::KeyEase::LINEAR, |key| key.ease);
            Command::SetClipKey {
                clip_id,
                property,
                at,
                value,
                ease,
            }
        };
        self.apply(command);
    }

    /// Takes every key off a field, leaving it its constant.
    pub fn clear_keys_on(&mut self, field: ClipField) {
        let Some(property) = Self::key_property_of(field) else {
            return;
        };
        let Some(clip_id) = self.sole_selection() else {
            return;
        };
        self.apply(Command::ClearClipKeys { clip_id, property });
    }

    /// Moves the playhead to this field's previous (-1) or next (+1) key.
    pub fn step_key(&mut self, field: ClipField, delta: i32) {
        let Some(property) = Self::key_property_of(field) else {
            return;
        };
        let Some((clip, at)) = self.key_point() else {
            return;
        };
        let (start, duration) = (clip.start, clip.duration);
        let (prev, next) = clip.keys_around(property, at);
        let Some(target) = (if delta < 0 { prev } else { next }) else {
            return;
        };
        self.seek((start + target * duration) as f32);
    }

    /// The audio tracks the Audio panel offers for the selection: one label
    /// per track of the selected clip's media when it has more than one,
    /// else nothing - a file with one track has nothing to choose.
    fn audio_track_labels(&self) -> Vec<SharedString> {
        let Some(item) = self
            .sole_selection()
            .and_then(|id| self.clip(&id))
            .and_then(|clip| self.project().media_by_id(&clip.media_id))
        else {
            return Vec::new();
        };
        if item.audio_tracks.len() < 2 {
            return Vec::new();
        }
        item.audio_tracks
            .iter()
            .enumerate()
            .map(|(position, track)| audio_track_label(track, position).into())
            .collect()
    }

    /// The selection, flattened for the inspector: exactly one clip or
    /// nothing.
    /// The selected transition, as the inspector's panel for it shows it.
    fn selected_transition_data(&self) -> crate::ui::SelectedTransitionData {
        let Some(clip) = self.selected_transition() else {
            return Default::default();
        };
        let transition = clip.transition_in.as_ref().expect("held to one");
        let name = Catalogue::builtin()
            .get(&transition.id)
            .map(|package| i18n::package_text(package.id(), "name", &package.manifest.effect.name))
            .unwrap_or_else(|| label_of(&transition.id));
        crate::ui::SelectedTransitionData {
            present: true,
            name: name.into(),
            duration: transition.duration as f32,
            into: clip.name.as_str().into(),
        }
    }

    fn selected(&self) -> SelectedClipData {
        // The one clip, or the first of a batch of titles: the panel shows
        // its values and writes to all of them; see `edit_targets`.
        let targets = self.edit_targets();
        let Some(clip) = targets.first().and_then(|id| self.clip(id)) else {
            return SelectedClipData::default();
        };
        // A keyed property shows what it is worth at the playhead, which is
        // what the knob edits.
        let at = place_in(clip, self.playhead);
        let text = clip.text.clone().unwrap_or_default();
        // Where the family sits in the picker's list; none when nothing on
        // this machine has it, which the picker shows in red.
        let family_row = self
            .font_families()
            .iter()
            .position(|family| family.eq_ignore_ascii_case(text.font_family.trim_matches('"')));
        let fill = colour_of(&text.color);
        let stroke = colour_of(&text.stroke_color);
        let plate = colour_of(&text.background);
        let analysis = clip.cutout.as_ref().and_then(|cutout| {
            self.cutout_jobs
                .get(&Self::analysis_key(&clip.media_id, cutout.subject))
                .copied()
        });
        SelectedClipData {
            present: true,
            batch: targets.len() as i32,
            frame_width: self.output_size().0 as i32,
            frame_height: self.output_size().1 as i32,
            id: clip.id.as_str().into(),
            name: clip.name.as_str().into(),
            kind: kind_of(clip),
            duration: clip.duration as f32,
            transition_id: clip
                .transition_in
                .as_ref()
                .map(|transition| transition.id.as_str())
                .unwrap_or_default()
                .into(),
            transition_name: clip
                .transition_in
                .as_ref()
                .and_then(|transition| Catalogue::builtin().get(&transition.id))
                .map(|package| {
                    i18n::package_text(package.id(), "name", &package.manifest.effect.name)
                })
                .unwrap_or_default()
                .into(),
            transition_duration: clip
                .transition_in
                .as_ref()
                .map(|transition| transition.duration as f32)
                .unwrap_or(0.5),
            scale: shown(clip, model::KeyProperty::Scale, at) as f32,
            offset_x: shown(clip, model::KeyProperty::OffsetX, at) as f32,
            offset_y: shown(clip, model::KeyProperty::OffsetY, at) as f32,
            rotation: shown(clip, model::KeyProperty::Rotation, at) as f32,
            stretch_x: clip.stretch_x as f32,
            stretch_y: clip.stretch_y as f32,
            opacity: shown(clip, model::KeyProperty::Opacity, at) as f32,
            volume: shown(clip, model::KeyProperty::Volume, at) as f32,
            audio_track: self
                .project()
                .media_by_id(&clip.media_id)
                .map_or(0, |item| item.audio_track_position(clip.audio_stream))
                as i32,
            speed: clip.speed as f32,
            preserve_pitch: clip.preserve_pitch,
            flip_h: clip.flip_h,
            flip_v: clip.flip_v,
            blend: concat_core::Blend::ALL
                .iter()
                .position(|mode| *mode == concat_core::Blend::parse(&clip.blend))
                .unwrap_or(0) as i32,
            crop_left: clip.crop.map(|crop| crop.left as f32).unwrap_or(0.0),
            crop_top: clip.crop.map(|crop| crop.top as f32).unwrap_or(0.0),
            crop_right: clip.crop.map(|crop| crop.right as f32).unwrap_or(0.0),
            crop_bottom: clip.crop.map(|crop| crop.bottom as f32).unwrap_or(0.0),
            fade_in: clip.fade_in as f32,
            fade_out: clip.fade_out as f32,
            content: text.content.as_str().into(),
            font_family: text.font_family.trim_matches('"').into(),
            family_row: family_row.unwrap_or(0) as i32,
            font_missing: family_row.is_none()
                && !text.font_family.trim_matches('"').trim().is_empty(),
            font_size: text.font_size as f32,
            font_weight: text.font_weight as f32,
            italic: text.italic,
            fill,
            fill_hex: hex_of(fill).into(),
            align: align_of(text.align),
            text_opacity: text.opacity as f32,
            stroke_width: text.stroke_width as f32,
            stroke,
            stroke_hex: hex_of(stroke).into(),
            stroke_opacity: f32::from(stroke.alpha()) / 255.0,
            shadow: text.shadow,
            plate,
            plate_hex: hex_of(plate).into(),
            plated: plate.alpha() > 0,
            plate_opacity: f32::from(plate.alpha()) / 255.0,
            plate_radius: text.background_radius as f32,
            plate_padding_x: text.background_padding_x as f32,
            plate_padding_y: text.background_padding_y as f32,
            line_height: text.line_height as f32,
            tracking: text.tracking as f32,
            text_width: text.max_width as f32,
            text_height: text.max_height as f32,
            cutout: match &clip.cutout {
                None => 0,
                Some(cutout) if cutout.mode == model::CutoutMode::Auto => 1,
                Some(_) => 2,
            },
            cutout_feather: clip
                .cutout
                .as_ref()
                .map(|cutout| cutout.feather as f32)
                .unwrap_or(model::DEFAULT_FEATHER as f32),
            cutout_strokes: clip
                .cutout
                .as_ref()
                .map(|cutout| cutout.strokes.len() as i32)
                .unwrap_or(0),
            cutout_subject: clip
                .cutout
                .as_ref()
                .map(|cutout| match cutout.subject {
                    model::Subject::Auto => 0,
                    model::Subject::Person => 1,
                    model::Subject::Object => 2,
                })
                .unwrap_or(0),
            cutout_progress: analysis.map(|(_, fraction)| fraction).unwrap_or(-1.0),
            cutout_fetching: analysis.is_some_and(|(fetching, _)| fetching),
            cutout_empty: self.cutout_empty(clip),
            football_analysed: self.football_analysed(&clip.id),
            football_progress: self
                .football_job
                .as_ref()
                .filter(|(_, id, _)| *id == clip.id)
                .map(|(_, _, fraction)| *fraction)
                .unwrap_or(-1.0),
            football_silhouettes: self
                .football_job
                .as_ref()
                .is_some_and(|(what, id, _)| *id == clip.id && what == "silhouettes"),
            football_picking: self.football_picking,
            football_chosen: self.football_chosen.map_or(-1, |id| id as i32),
            football_ball: self.football_chosen == Some(concat_vision::football::BALL),
            football_shade: self.football_marked(clip, model::MarkStyle::Shade),
            football_spotlight: self.football_marked(clip, model::MarkStyle::Spotlight),
            football_ring: self.football_marked(clip, model::MarkStyle::Ring),
            football_erase: self.football_marked(clip, model::MarkStyle::Erase),
            football_isolate: self.football_marked(clip, model::MarkStyle::Isolate),
            football_marks: clip.football.len() as i32,
            draw_tool: self.draw_tool.map_or(-1, |index| index as i32),
            draw_dashed: self.draw_style.dashed,
            draw_thickness: self.draw_style.thickness as f32,
            draw_hold: self.draw_style.hold as f32,
            draw_reveal: self.draw_style.reveal as f32,
            draw_bend: self.draw_style.bend as f32,
            draw_freeze: self.draw_style.freeze as f32,
            draw_colour: Self::MARK_COLOURS
                .iter()
                .position(|colour| *colour == self.draw_style.colour)
                .map_or(-1, |index| index as i32),
            drawings: clip.drawings.len() as i32,
            draw_points: self
                .draw_draft
                .as_ref()
                .filter(|(id, _, _)| *id == clip.id)
                .map_or(0, |(_, anchors, _)| anchors.len() as i32),
            football_arrow: self.football_marked(clip, model::MarkStyle::Arrow),
            football_frame: self.football_marked(clip, model::MarkStyle::Frame),
            football_trail: self.football_marked(clip, model::MarkStyle::Trail),
            football_lens: self.football_marked(clip, model::MarkStyle::Lens),
            football_distance: self.football_marked(clip, model::MarkStyle::Distance),
            football_speed: self.football_marked(clip, model::MarkStyle::Speed),
            football_last_line: self.football_marked(clip, model::MarkStyle::LastLine),
            football_lanes: self.football_marked(clip, model::MarkStyle::Lanes),
            football_marks_here: self.football_chosen.map_or(0, |track| {
                clip.football
                    .iter()
                    .filter(|mark| mark.track == track)
                    .count()
            }) as i32,
            football_size: self.football_knob(clip, |mark| mark.size),
            football_thickness: self.football_knob(clip, |mark| mark.thickness),
            football_squash: self.football_knob(clip, |mark| mark.squash),
            football_width: self.football_knob(clip, |mark| mark.width),
            football_strength: self
                .football_chosen
                .and_then(|track| clip.football.iter().find(|mark| mark.track == track))
                .map_or(0.8, |mark| mark.strength as f32),
            football_colour: self
                .football_chosen
                .and_then(|track| clip.football.iter().find(|mark| mark.track == track))
                .and_then(|mark| Self::MARK_COLOURS.iter().position(|c| *c == mark.colour))
                .map_or(-1, |index| index as i32),
            enhance_progress: self
                .enhance_jobs
                .get(&clip.id)
                .map(|(_, fraction)| *fraction)
                .unwrap_or(-1.0),
            enhance_fetching: self
                .enhance_jobs
                .get(&clip.id)
                .is_some_and(|(fetching, _)| *fetching),
            reverse_progress: self.reverse_jobs.get(&clip.id).copied().unwrap_or(-1.0),
            enhance_linked: clip.enhanced.is_some(),
            enhance_on: clip.enhanced.as_ref().is_some_and(|link| link.on)
                || self.enhance_jobs.contains_key(&clip.id),
        }
    }

    /// A knob of the chosen track's first mark on `clip`, or its default.
    fn football_knob(&self, clip: &Clip, knob: impl Fn(&model::FootballMark) -> f64) -> f32 {
        self.football_chosen
            .and_then(|track| clip.football.iter().find(|mark| mark.track == track))
            .map_or(1.0, |mark| knob(mark) as f32)
    }

    /// Whether the chosen track wears a mark of `style` on `clip`.
    fn football_marked(&self, clip: &Clip, style: model::MarkStyle) -> bool {
        self.football_chosen.is_some_and(|track| {
            clip.football
                .iter()
                .any(|mark| mark.track == track && mark.style == style)
        })
    }

    /// The source instant of `clip` under the playhead, held to the clip.
    fn source_at_playhead(&self, clip: &Clip) -> f64 {
        let along = (f64::from(self.playhead) - clip.start).clamp(0.0, clip.duration);
        clip.source_start + along * clip.speed
    }

    /// Whether the cutout model found nothing at the playhead's frame of
    /// `clip`, so the picture is showing as shot. Only an automatic cutout
    /// says so: a custom one is whatever was painted.
    fn cutout_empty(&self, clip: &Clip) -> bool {
        let Some(cutout) = clip.cutout.as_ref() else {
            return false;
        };
        if cutout.mode != model::CutoutMode::Auto {
            return false;
        }
        let (Some(session), Some(media)) = (
            self.session.as_ref(),
            self.project().media_by_id(&clip.media_id),
        ) else {
            return false;
        };
        let project = std::path::Path::new(session.path());
        let store = concat_vision::MaskStore::open(&concat_vision::mask_dir(
            project,
            &media.path,
            cutout.subject,
        ));
        store
            .mask_at(self.source_at_playhead(clip))
            .is_some_and(|mask| mask.is_blank())
    }

    /// Starts reading the first smart stroke whose region is not there
    /// yet, unless one is being read. Called after every change, and by
    /// the finished job for whatever is next.
    pub fn ensure_regions(&mut self) {
        if self.region_job.is_some() || self.host.brushes.is_busy() {
            return;
        }
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let project = std::path::PathBuf::from(session.path());
        let mut next: Option<(String, RegionRequest)> = None;
        'clips: for clip in &self.timeline().clips {
            let Some(cutout) = clip.cutout.as_ref() else {
                continue;
            };
            if cutout.mode != model::CutoutMode::Custom || !clip.kind.is_visual() {
                continue;
            }
            let Some(media) = self.project().media_by_id(&clip.media_id) else {
                continue;
            };
            // The source the clip shows, as the mask analysis reckons it.
            let range = (
                clip.source_start,
                clip.source_start + clip.duration * clip.speed.max(0.0625),
            );
            for stroke in &cutout.strokes {
                let request = RegionRequest {
                    project: project.clone(),
                    media_path: media.path.clone(),
                    media_size: (media.width.unwrap_or(0), media.height.unwrap_or(0)),
                    still: media.kind == model::MediaKind::Image,
                    subject: cutout.subject,
                    ranges: vec![range],
                    stroke: stroke.clone(),
                };
                if request.outstanding() {
                    next = Some((Self::analysis_key(&media.id, cutout.subject), request));
                    break 'clips;
                }
            }
        }
        let Some((key, request)) = next else {
            return;
        };
        self.region_job = Some(key.clone());
        self.cutout_jobs.entry(key.clone()).or_insert((false, 0.0));
        let brushes = Arc::clone(&self.host.brushes);
        let epoch = crate::host::project_epoch();
        spawn_in_project(
            move || {
                let reporting = key.clone();
                let result = brushes.read(&request, &mut |progress| {
                    let now = match progress {
                        concat_host::cutout::Progress::Fetching { received, total } => {
                            (true, received as f32 / total.max(1) as f32)
                        }
                        concat_host::cutout::Progress::Analysing(fraction) => (false, fraction),
                    };
                    let key = reporting.clone();
                    on_ui_in_project(epoch, move |studio, _, _| {
                        if let Some(held) = studio.cutout_jobs.get_mut(&key) {
                            *held = now;
                        }
                    });
                });
                (key, result)
            },
            |studio, _, _, (key, result)| {
                studio.region_job = None;
                studio.cutout_jobs.remove(&key);
                studio.pending_stroke = None;
                match result {
                    Ok(()) => {
                        studio.request_preview();
                        studio.ensure_regions();
                    }
                    Err(error) if error.contains("cancelled") => {}
                    Err(error) => studio.notify(&tf("studio.smartBrush", &[&error]), true),
                }
            },
        );
    }

    /// The menus, the dialogs, the bin and the engine lists.
    pub fn publish_chrome(&self, app: &App, models: &Models) {
        let editor = app.global::<Editor>();

        // The language, when it changed: one property the whole tree
        // reads, written only when it differs so nothing re-evaluates
        // for nothing.
        let words = app.global::<I18n>();
        let lang = SharedString::from(i18n::current());
        if words.get_lang() != lang {
            words.set_lang(lang);
        }

        // The catalogue's shelves. The groups are built in and never change;
        // the entries are whatever each library's search, shelf and star let
        // through, so they are rebuilt here and `sync` makes an unchanged
        // one a no-op.
        let starred = &self.prefs.favourites;
        let stamp = ShelfStamp {
            catalogue: std::ptr::from_ref(Catalogue::builtin()) as usize,
            cards: self.cards_drawn,
            lang: i18n::current(),
            views: self
                .library
                .iter()
                .map(|view| {
                    (
                        view.query.clone(),
                        view.group,
                        view.favourites,
                        view.category.clone(),
                    )
                })
                .collect(),
            favourites: starred.clone(),
        };
        if self.shelf_stamp.borrow().as_ref() != Some(&stamp) {
            let (groups, entries) = shelves(
                SHELF_KINDS[0],
                &self.library[0],
                starred,
                &self.look_art,
                &self.card_paths,
            );
            sync(&models.filter_groups, groups);
            sync(&models.catalogue_filters, entries);
            let (groups, entries) = shelves(
                SHELF_KINDS[1],
                &self.library[1],
                starred,
                &self.look_art,
                &self.card_paths,
            );
            sync(&models.effect_groups, groups);
            sync(&models.catalogue_effects, entries);
            let (groups, entries) = shelves(
                SHELF_KINDS[2],
                &self.library[2],
                starred,
                &self.look_art,
                &self.card_paths,
            );
            sync(&models.audio_groups, groups);
            sync(&models.catalogue_audio, entries);
            let (groups, entries) = shelves(
                SHELF_KINDS[3],
                &self.library[3],
                starred,
                &self.look_art,
                &self.card_paths,
            );
            sync(&models.transition_groups, groups);
            sync(&models.catalogue_transitions, entries);
            *self.shelf_stamp.borrow_mut() = Some(stamp);
        }
        sync(
            &models.library_views,
            self.library
                .iter()
                .map(|view| LibraryViewData {
                    query: view.query.as_str().into(),
                    group: view.group,
                    favourites: view.favourites,
                    category: view.category.as_str().into(),
                })
                .collect(),
        );

        app.set_on_start(self.on_start);
        app.set_project_name(self.project_name.as_str().into());
        let (home, score, away) = scoreboard(&self.project_name).unwrap_or_default();
        app.set_score_home(home.into());
        app.set_score(score.into());
        app.set_score_away(away.into());
        app.set_project_status(
            if self.dirty {
                t("common.unsavedChanges")
            } else {
                t("common.saved")
            }
            .into(),
        );
        app.set_toast(ToastData {
            token: self.toast.token,
            message: self.toast.message.as_str().into(),
            failed: self.toast.failed,
        });
        app.set_start(self.start.data());
        sync(
            &models.recents,
            self.recents
                .iter()
                .map(|project| RecentProjectData {
                    path: project.path.as_str().into(),
                    name: project.name.as_str().into(),
                    detail: format!(
                        "{} x {} · {:.2} fps",
                        project.width,
                        project.height,
                        project.rate_num as f32 / project.rate_den.max(1) as f32
                    )
                    .into(),
                    when: when_phrase(project.opened_at).into(),
                    poster: self.posters.get(&project.path).cloned().unwrap_or_default(),
                })
                .collect(),
        );

        // The Text page's presets: the look each card draws its name in.
        // All of them for the phone's one shelf; on the desktop, Styles
        // has the Latin ones and Languages, under it, the rest.
        let cards: Vec<TextPresetData> = self
            .text_presets
            .iter()
            .map(|preset| {
                let plate = colour_of(&preset.style.background);
                TextPresetData {
                    id: preset.id.as_str().into(),
                    name: preset.name.as_str().into(),
                    language: preset.language.as_str().into(),
                    sample: preset.sample.as_str().into(),
                    family: preset.style.font_family.trim_matches('"').into(),
                    weight: preset.style.font_weight.round() as i32,
                    italic: preset.style.italic,
                    fill: colour_of(&preset.style.color),
                    plate,
                    plated: plate.alpha() > 0,
                    stroke: colour_of(&preset.style.stroke_color),
                    stroke_width: preset.style.stroke_width as f32,
                    align: align_of(preset.style.align),
                }
            })
            .collect();
        let latin = |card: &TextPresetData| card.language.eq_ignore_ascii_case("latin");
        sync(
            &models.style_presets,
            cards.iter().filter(|card| latin(card)).cloned().collect(),
        );
        sync(
            &models.language_presets,
            cards.iter().filter(|card| !latin(card)).cloned().collect(),
        );
        sync(&models.text_presets, cards);

        // The bin.
        // The Media shelves count the imports; what the editor made is
        // counted on its own shelf under Generated. See MediaBin::shows.
        let items = &self.project().media;
        sync(&models.media, self.media.rows(self));
        let imported = |kind: Option<model::MediaKind>| {
            items
                .iter()
                .filter(|item| item.origin.is_none() && kind.is_none_or(|kind| item.kind == kind))
                .count() as i32
        };
        editor.set_media_count_all(imported(None));
        editor.set_media_count_video(imported(Some(model::MediaKind::Video)));
        editor.set_media_count_audio(imported(Some(model::MediaKind::Audio)));
        editor.set_media_count_images(imported(Some(model::MediaKind::Image)));
        editor.set_media_count_speech(
            items
                .iter()
                .filter(|item| item.origin == Some(model::MediaOrigin::Speech))
                .count() as i32,
        );
        editor.set_media_count_processed(
            items
                .iter()
                .filter(|item| item.origin == Some(model::MediaOrigin::Processed))
                .count() as i32,
        );
        editor.set_media_selected_count(self.media.selected.len() as i32);
        editor.set_media_filter(self.media.filter);
        editor.set_media_jump_token(self.media_jump);
        editor.set_importing(false);

        let (width, height) = self.output_size();
        editor.set_output_width(width as i32);
        editor.set_output_height(height as i32);
        editor.set_ratio_index(
            OUTPUTS
                .iter()
                .position(|size| *size == (width as i32, height as i32))
                .map_or(-1, |index| index as i32),
        );
        editor.set_quality_index(self.quality_of() as i32);

        // The Details panel, and the sheet its Modify button opens. The
        // frame, rate and duration are the active timeline's, and the panel
        // says so by name; the name and folder are the project's.
        let folder: SharedString = self
            .session
            .as_ref()
            .map(|session| session.path().to_owned())
            .unwrap_or_else(|| "—".to_owned())
            .into();
        let timeline_name: SharedString = self.timeline().name.as_str().into();
        editor.set_project_name(self.project_name.as_str().into());
        editor.set_project_folder(folder.clone());
        editor.set_timeline_name(timeline_name.clone());
        editor.set_project_output(format!("{width} × {height}").into());
        editor.set_project_rate(format!("{:.2} fps", self.frame_rate()).into());
        editor.set_project_duration(frames_timecode(self.duration(), self.frame_rate()).into());
        editor.set_count_media(self.project().media.len() as i32);
        editor.set_count_tracks(self.timeline().tracks.len() as i32);
        editor.set_count_clips(self.timeline().clips.len() as i32);
        app.set_project_sheet(self.project_sheet.data(self));

        let rows = self.menu();
        editor.set_menu_height(Self::menu_height(&rows));
        sync(&models.menu, rows);
        editor.set_menu_token(self.menu_token);

        app.set_export(self.export.data(self));
        app.set_settings(self.settings.data(self));
        sync(&models.transcribers, self.settings.transcriber_rows());
        sync(&models.voices, self.settings.voice_rows());
        sync(&models.versions, self.settings.version_rows());
        sync(&models.version_details, self.settings.version_details());
        app.set_relink(self.relink.data());
        app.set_confirm(self.confirm_data());

        // The speech sheets, and the lists they choose from.
        let transcribers = installed(&self.settings.transcribers);
        sync(
            &models.caption_models,
            transcribers
                .iter()
                .map(|model| SharedString::from(model.name.as_str()))
                .collect(),
        );
        app.set_captions(self.captions.data(self));
        let voices = installed(&self.settings.voices);
        sync(
            &models.speech_models,
            voices
                .iter()
                .map(|model| SharedString::from(model.name.as_str()))
                .collect(),
        );
        sync(
            &models.speech_model_details,
            self.speech.model_detail_rows(self),
        );
        sync(&models.speakers, self.speech.speaker_rows());
        sync(&models.speaker_details, self.speech.speaker_detail_rows());
        sync(&models.speech_samples, self.speech.sample_rows(self));
        sync(
            &models.speech_sample_waves,
            self.speech.sample_wave_rows(self),
        );
        sync(
            &models.speech_sample_details,
            self.speech.sample_detail_rows(self),
        );
        app.set_speech(self.speech.data(self));
        self.publish_source(app);
        editor.set_recording(self.voiceover.recording());

        let bar = self.menu_bar();
        app.set_app_menu_height(Self::menu_height(&bar));
        sync(&models.bar, bar);
        app.set_app_menu_token(self.menu_bar_token);
        app.set_open_menu(self.open_menu);
    }

    /// Asks the workers for anything the launch screen or the bin is
    /// missing. Separate from `publish` because it mutates.
    pub fn refresh_art(&mut self) {
        // Called after every callback, the pointer stream included. The
        // scan over every media item - stats for proxies, JPEGs read back,
        // decodes queued - runs when the document, the screen or what is
        // pending changed, and is one comparison otherwise. The lanes'
        // cell strips follow the view, which moves on that stream, so
        // they are asked for every time; that walk is hash lookups
        // (audit 2026-09-23, #11).
        let stamp = (
            self.revision,
            self.on_start,
            self.art_pending.len() + self.posters_pending.len(),
        );
        if self.art_stamp == Some(stamp) {
            if !self.on_start {
                self.request_window_art();
            }
            return;
        }
        self.art_stamp = Some(stamp);
        if self.on_start {
            self.request_posters();
        } else {
            self.assign_media_rows();
            self.request_media_art();
        }
    }

    /// The right-click menu: the bin card's when it was opened on one,
    /// else the clip's.
    fn menu(&self) -> Vec<MenuItemData> {
        if let Some(id) = self.menu_media.as_deref() {
            return self.media_menu(id);
        }
        let Some(clip) = self.menu_target.as_ref().and_then(|id| self.clip(id)) else {
            return Vec::new();
        };
        let locked = self.locked(&clip.track_id);
        let playhead = f64::from(self.playhead);
        let straddled = clip.start < playhead && playhead < clip.start + clip.duration;

        let action =
            |id: &str, label: String, glyph: Glyph, shortcut: &str, enabled: bool| MenuItemData {
                id: id.into(),
                label: label.into(),
                kind: MenuRow::Action,
                glyph,
                shortcut: shortcut.into(),
                enabled,
                danger: false,
                checkable: false,
                checked: false,
            };
        let check = |id: &str, label: &str, shortcut: &str, on: bool, enabled: bool| MenuItemData {
            id: id.into(),
            label: label.into(),
            kind: MenuRow::Action,
            glyph: Glyph::None,
            shortcut: shortcut.into(),
            enabled,
            danger: false,
            checkable: true,
            checked: on,
        };
        let rule = || MenuItemData {
            kind: MenuRow::Separator,
            ..Default::default()
        };

        let mut rows = vec![
            action(
                "copy",
                t("common.copy"),
                Glyph::Copy,
                &platform::keys(&["Control", "C"]),
                true,
            ),
            action(
                "duplicate",
                t("studio.duplicate"),
                Glyph::Plus,
                &platform::keys(&["Control", "D"]),
                !locked,
            ),
            action(
                "paste",
                t("studio.paste"),
                Glyph::Plus,
                &platform::keys(&["Control", "V"]),
                !self.clipboard.is_empty(),
            ),
            rule(),
            action(
                "split",
                t("studio.splitAtPlayhead"),
                Glyph::Split,
                &platform::keys(&["Control", "B"]),
                straddled && !locked,
            ),
            action(
                "freeze",
                "Freeze frame".into(),
                Glyph::Frame,
                "F",
                straddled
                    && !locked
                    && (clip.kind == model::ClipKind::Video || clip.kind == model::ClipKind::Image),
            ),
            // The picture restored and enlarged by the model: the row opens
            // the clip's Enhance in the inspector, whose switch runs it.
            action(
                "enhance",
                t("common.enhance"),
                Glyph::Sparkle,
                "",
                !locked
                    && (clip.kind == model::ClipKind::Video || clip.kind == model::ClipKind::Image),
            ),
            rule(),
        ];
        // One slot, two verbs: the sound is either on its picture or off it.
        // Shown on every video clip, so the verb is where a person looks for
        // it, and greyed rather than gone when the file has no sound.
        let (can_detach, can_reattach) = self.sound_placement(clip);
        if can_reattach {
            rows.push(action(
                "reattach",
                t("studio.reattachAudio"),
                Glyph::Merge,
                "",
                !locked,
            ));
            rows.push(rule());
        } else if clip.kind == model::ClipKind::Video {
            rows.push(action(
                "detach",
                t("studio.detachAudio"),
                Glyph::Waveform,
                "",
                !locked && can_detach && self.clip_has_sound(clip),
            ));
            rows.push(rule());
        }
        // The clip's sound, written out as it plays, as a file of its own
        // in the bin. For anything with sound in it; a clip whose track is
        // muted still renders, since the file is the clip's and not the
        // mix's.
        if clip.kind == model::ClipKind::Audio || clip.kind == model::ClipKind::Video {
            rows.push(action(
                "export-audio",
                t("studio.exportAudio"),
                Glyph::Waveform,
                "",
                self.clip_has_sound(clip),
            ));
            rows.push(rule());
        }
        let audible = clip.kind != model::ClipKind::Image;
        rows.push(check(
            "mute",
            &t("studio.mute"),
            "M",
            clip.volume <= 0.0,
            !locked && audible,
        ));
        rows.push(check("lock", &t("studio.lockTrack"), "", locked, true));
        rows.push(rule());
        rows.push(MenuItemData {
            id: "delete".into(),
            label: t("common.delete").into(),
            kind: MenuRow::Action,
            glyph: Glyph::Trash,
            shortcut: platform::delete_key(false).into(),
            enabled: !locked,
            danger: true,
            checkable: false,
            checked: false,
        });
        // One Delete, not a Delete and a Ripple Delete: whether a delete
        // closes the gap is the tray's magnetic switch, a setting of the
        // timeline rather than a choice to make at every clip.
        rows
    }

    /// The right-click menu for a card in the bin: the file's verbs, and
    /// the one attribute a file has that every clip cut from it inherits.
    ///
    /// A picture's colour range is the media's rather than any clip's -
    /// the fix for a file that lies about its levels is a fact about the
    /// file - so it is set here, off the card, the way Resolve keeps Clip
    /// Attributes off the media pool, and not in a clip's inspector where
    /// it read as a property of the cut. Auto is the file's tag, and the
    /// codec's convention where there is none; see
    /// `concat_media::ColorRange::implied`.
    /// https://github.com/jub0t/Concat/issues/103
    fn media_menu(&self, id: &str) -> Vec<MenuItemData> {
        let Some(item) = self.project().media_by_id(id) else {
            return Vec::new();
        };
        let action = |id: &str, label: String, glyph: Glyph, danger: bool| MenuItemData {
            id: id.into(),
            label: label.into(),
            kind: MenuRow::Action,
            glyph,
            shortcut: "".into(),
            enabled: true,
            danger,
            checkable: false,
            checked: false,
        };
        // The levels go where a shortcut would: the number is what the
        // word means, and a person who knows one knows the other.
        let check = |id: &str, label: String, levels: &str, on: bool| MenuItemData {
            id: id.into(),
            label: label.into(),
            kind: MenuRow::Action,
            glyph: Glyph::None,
            shortcut: levels.into(),
            enabled: true,
            danger: false,
            checkable: true,
            checked: on,
        };
        let rule = || MenuItemData {
            kind: MenuRow::Separator,
            ..Default::default()
        };
        let mut rows = vec![
            action("add", t("studio.addAtPlayhead"), Glyph::Plus, false),
            rule(),
        ];
        if item.kind != model::MediaKind::Audio {
            let range = item.color_range;
            rows.push(MenuItemData {
                label: t("common.colourRange").to_uppercase().into(),
                kind: MenuRow::Label,
                ..Default::default()
            });
            rows.push(check("range-auto", t("common.auto"), "", range.is_none()));
            rows.push(check(
                "range-limited",
                t("common.limited"),
                "16-235",
                range == Some(model::ColorRange::Limited),
            ));
            rows.push(check(
                "range-full",
                t("common.full"),
                "0-255",
                range == Some(model::ColorRange::Full),
            ));
            rows.push(rule());
        }
        rows.push(action("remove", t("common.remove"), Glyph::Trash, true));
        rows
    }

    /// A row of a bin card's menu, for the media it was opened on.
    pub fn media_action(&mut self, id: &str, action: &str) {
        log::info!("media menu: {action} on {id}");
        let Some(row) = self.media.row_of(id) else {
            return;
        };
        if self.project().media_by_id(id).is_none() {
            return;
        }
        let range = |range| Command::SetMediaColorRange {
            media_id: id.to_owned(),
            range,
        };
        match action {
            "add" => self.place_at_playhead(&format!("media:{row}")),
            "remove" => self.handle(crate::panes::Msg::Media(
                crate::panes::media_bin::MediaMsg::Remove(row),
            )),
            "range-auto" => {
                self.apply(range(None));
            }
            "range-limited" => {
                self.apply(range(Some(model::ColorRange::Limited)));
            }
            "range-full" => {
                self.apply(range(Some(model::ColorRange::Full)));
            }
            _ => {}
        }
    }

    pub fn menu_height(rows: &[MenuItemData]) -> f32 {
        let metrics = |kind: MenuRow| match kind {
            MenuRow::Action => 26.0,
            MenuRow::Label => 24.0,
            MenuRow::Separator => 9.0,
        };
        rows.iter().map(|row| metrics(row.kind)).sum::<f32>() + 12.0
    }

    fn menu_bar(&self) -> Vec<MenuItemData> {
        let row =
            |id: &str, label: String, glyph: Glyph, shortcut: &str, enabled: bool| MenuItemData {
                id: id.into(),
                label: label.into(),
                kind: MenuRow::Action,
                glyph,
                shortcut: shortcut.into(),
                enabled,
                danger: false,
                checkable: false,
                checked: false,
            };
        let rule = || MenuItemData {
            kind: MenuRow::Separator,
            ..Default::default()
        };
        let check = |id: &str, label: &str, on: bool| MenuItemData {
            id: id.into(),
            label: label.into(),
            kind: MenuRow::Action,
            glyph: Glyph::None,
            shortcut: "".into(),
            enabled: true,
            danger: false,
            checkable: true,
            checked: on,
        };
        let selected = self.selection.len();
        let playhead = f64::from(self.playhead);
        let straddled = self.timeline().clips.iter().any(|clip| {
            clip.start + f64::from(MIN_DURATION) < playhead
                && playhead < clip.start + clip.duration - f64::from(MIN_DURATION)
        });
        let (can_undo, can_redo) = self.session.as_ref().map_or((false, false), |session| {
            (session.can_undo(), session.can_redo())
        });
        let has_selection_media = !self.media.selected.is_empty();

        match self.open_menu {
            0 => vec![
                row(
                    "add-selected",
                    t("studio.addSelectedToTimeline"),
                    Glyph::Plus,
                    "",
                    has_selection_media,
                ),
                row(
                    "open",
                    t("studio.openProject"),
                    Glyph::Import,
                    &platform::keys(&["Control", "O"]),
                    true,
                ),
                row(
                    "import",
                    t("studio.importMedia"),
                    Glyph::Import,
                    &platform::keys(&["Control", "I"]),
                    true,
                ),
                row(
                    "save",
                    t("studio.save"),
                    Glyph::Import,
                    &platform::keys(&["Control", "S"]),
                    true,
                ),
                row(
                    "export",
                    t("studio.export"),
                    Glyph::Export,
                    "",
                    !self.timeline().clips.is_empty(),
                ),
                row(
                    "template",
                    t("studio.saveAsTemplate"),
                    Glyph::Slot,
                    "",
                    true,
                ),
                row("speech", t("studio.textToSpeech"), Glyph::Volume, "", true),
                row(
                    "clear-cache",
                    t("studio.clearProjectCache"),
                    Glyph::None,
                    "",
                    true,
                ),
                rule(),
                row(
                    "settings",
                    t("studio.settings"),
                    Glyph::Settings,
                    &platform::keys(&["Control", ","]),
                    true,
                ),
                rule(),
                // The last row is each platform's own word for leaving:
                // Close Window on a Mac, where ⌘W is it; Exit on Windows,
                // where Alt+F4 is; Quit on Linux, with Ctrl+Q. Off the Mac
                // Ctrl+W closes the document instead, which is the project.
                row(
                    "close-project",
                    t("studio.closeProject"),
                    Glyph::Import,
                    &if platform::MACOS {
                        String::new()
                    } else {
                        platform::keys(&["Control", "W"])
                    },
                    true,
                ),
                MenuItemData {
                    id: "close-window".into(),
                    label: if platform::MACOS {
                        t("studio.closeWindow")
                    } else if cfg!(windows) {
                        t("studio.exit")
                    } else {
                        t("studio.quit")
                    }
                    .into(),
                    kind: MenuRow::Action,
                    glyph: Glyph::Close,
                    shortcut: if platform::MACOS {
                        platform::keys(&["Control", "W"])
                    } else if cfg!(windows) {
                        platform::keys(&["Alt", "F4"])
                    } else {
                        platform::keys(&["Control", "Q"])
                    }
                    .into(),
                    enabled: true,
                    danger: true,
                    checkable: false,
                    checked: false,
                },
            ],
            1 => vec![
                row(
                    "undo",
                    t("studio.undo"),
                    Glyph::Undo,
                    &platform::keys(&["Control", "Z"]),
                    can_undo,
                ),
                row(
                    "redo",
                    t("studio.redo"),
                    Glyph::Redo,
                    &platform::keys(&["Control", "Shift", "Z"]),
                    can_redo,
                ),
                rule(),
                row(
                    "split",
                    t("studio.splitAtPlayhead"),
                    Glyph::Razor,
                    &platform::keys(&["Control", "B"]),
                    straddled,
                ),
                MenuItemData {
                    id: "delete".into(),
                    label: if selected > 1 {
                        tf("studio.deleteClips", &[&selected])
                    } else {
                        t("studio.deleteClip")
                    }
                    .into(),
                    kind: MenuRow::Action,
                    glyph: Glyph::Trash,
                    shortcut: platform::delete_key(false).into(),
                    enabled: selected > 0,
                    danger: true,
                    checkable: false,
                    checked: false,
                },
                rule(),
                MenuItemData {
                    id: "snap".into(),
                    label: t("common.snapToEdges").into(),
                    kind: MenuRow::Action,
                    glyph: Glyph::None,
                    shortcut: "N".into(),
                    enabled: true,
                    danger: false,
                    checkable: true,
                    checked: self.lanes.snap,
                },
                MenuItemData {
                    id: "magnetic".into(),
                    label: t("common.magneticTimeline").into(),
                    kind: MenuRow::Action,
                    glyph: Glyph::None,
                    shortcut: "".into(),
                    enabled: true,
                    danger: false,
                    checkable: true,
                    checked: self.prefs.magnetic,
                },
            ],
            2 => vec![
                row("zoom-in", t("common.zoomIn"), Glyph::Plus, "+", true),
                row("zoom-out", t("common.zoomOut"), Glyph::Minus, "-", true),
                rule(),
                check("sort-added", "Sort by: Added", self.media.sort == 0),
                check("sort-name", "Sort by: Name", self.media.sort == 1),
                check("sort-kind", "Sort by: Type", self.media.sort == 2),
                rule(),
                row(
                    "start",
                    t("common.goToStart"),
                    Glyph::SkipBack,
                    "Home",
                    true,
                ),
                row("end", t("common.goToEnd"), Glyph::SkipForward, "End", true),
                rule(),
                check(
                    "workspace-analysis",
                    &t("bar.workspace"),
                    self.analysis_workspace(),
                ),
            ],
            3 => self.montage_menu(),
            _ => Vec::new(),
        }
    }

    /// The clip's track flags, as one command per flag that changed, plus
    /// the lock, which is the window's.
    pub fn track_flags(&mut self, row: i32, visible: bool, muted: bool, locked: bool) {
        let Some(track) = self.row_track(row).cloned() else {
            return;
        };
        let mut commands = Vec::new();
        if track.visible != visible {
            commands.push(Command::SetTrackFlag {
                track_id: track.id.clone(),
                flag: TrackFlag::Visible,
                value: visible,
            });
        }
        if track.muted != muted {
            commands.push(Command::SetTrackFlag {
                track_id: track.id.clone(),
                flag: TrackFlag::Muted,
                value: muted,
            });
        }
        let view = self.lanes.lane_view.entry(track.id.clone()).or_default();
        view.locked = locked;
        if locked {
            let doomed: Vec<String> = self
                .timeline()
                .clips
                .iter()
                .filter(|clip| clip.track_id == track.id)
                .map(|clip| clip.id.clone())
                .collect();
            self.selection.retain(|held| !doomed.contains(held));
        }
        match commands.len() {
            0 => {}
            1 => {
                self.apply(commands.remove(0));
            }
            _ => {
                self.apply(Command::Batch { commands });
            }
        }
    }

    pub fn set_lane_size(&mut self, row: i32, size: TrackSize) {
        if let Some(id) = self.row_track(row).map(|track| track.id.clone()) {
            self.lanes.lane_view.entry(id).or_default().size = size;
        }
    }

    /// A quarter turn clockwise on every selected picture, at the playhead:
    /// a keyed rotation gets a key there and a constant one turns whole, as
    /// the stage's own rotate grip writes it. One undo step for the lot.
    pub fn rotate_selected(&mut self) {
        self.flush_commit();
        let playhead = self.playhead;
        let ids: Vec<String> = self
            .selection
            .iter()
            .filter(|id| {
                self.clip(id).is_some_and(|clip| {
                    clip.kind != model::ClipKind::Audio && !self.locked(&clip.track_id)
                })
            })
            .cloned()
            .collect();
        let mut commands = Vec::new();
        for id in ids {
            let Some(before) = self.clip(&id).cloned() else {
                continue;
            };
            let mut after = before.clone();
            let at = place_in(&after, playhead);
            // Kept in the inspector's range, wrapping rather than stopping,
            // as the grip does.
            let next = (shown(&after, model::KeyProperty::Rotation, at) + 90.0 + 180.0)
                .rem_euclid(360.0)
                - 180.0;
            write_keyable(&mut after, model::KeyProperty::Rotation, next, at);
            if after.rotation != before.rotation {
                commands.push(Command::SetClipTransform {
                    clip_id: id.clone(),
                    scale: Some(after.scale),
                    offset_x: Some(after.offset_x),
                    offset_y: Some(after.offset_y),
                    rotation: Some(after.rotation),
                    stretch_x: Some(after.stretch_x),
                    stretch_y: Some(after.stretch_y),
                });
            }
            commands.extend(key_commands(&id, &before, &after));
        }
        match commands.len() {
            0 => {}
            1 => {
                self.apply(commands.remove(0));
            }
            _ => {
                self.apply(Command::Batch { commands });
            }
        }
    }

    pub fn toggle_flip_h(&mut self) {
        if self.selection.is_empty() {
            return;
        }
        let commands: Vec<Command> = self
            .selection
            .iter()
            .filter_map(|id| {
                self.clip(id).map(|clip| Command::UpdateClip {
                    clip_id: id.clone(),
                    patch: ClipPatch {
                        flip_h: Some(!clip.flip_h),
                        ..Default::default()
                    },
                })
            })
            .collect();
        if !commands.is_empty() {
            self.apply(Command::Batch { commands });
        }
    }

    pub fn toggle_lock(&mut self, track_id: &str) {
        let view = self.lanes.lane_view.entry(track_id.to_owned()).or_default();
        view.locked = !view.locked;
        if view.locked {
            let doomed: Vec<String> = self
                .timeline()
                .clips
                .iter()
                .filter(|clip| clip.track_id == track_id)
                .map(|clip| clip.id.clone())
                .collect();
            self.selection.retain(|held| !doomed.contains(held));
        }
    }

    /// Changes the active timeline's output size from the monitor's picker.
    /// An edit like any other - undoable, and this timeline's alone.
    pub fn set_output(&mut self, index: usize) {
        let (width, height) = OUTPUTS[index.min(OUTPUTS.len() - 1)];
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let video = model::VideoSettings {
            width: width as u32,
            height: height as u32,
            ..session.video()
        };
        let timeline_id = self.project().active_timeline_id.clone();
        self.apply(Command::SetTimelineVideo { timeline_id, video });
        self.request_preview();
    }

    // ── the keyboard, and the menu's verbs ──

    /// A chord from the window's key table; see `Editor.shortcut`. The view
    /// chords - zoom, the playhead's ends, the sheets - go through the app
    /// menu's own handler in lib.rs, so a key and the row that advertises
    /// it are one thing.
    pub fn shortcut(&mut self, action: &str) {
        log::info!("shortcut: {action}");
        match action {
            "split" => {
                let at = self.playhead;
                self.split_at(at, false);
            }
            "split-selected" => {
                let at = self.playhead;
                self.split_at(at, true);
            }
            // B: the razor, on and off.
            "split-tool" => {
                self.lanes.tool = if self.lanes.tool == TimelineTool::Split {
                    TimelineTool::Select
                } else {
                    TimelineTool::Split
                };
            }
            "select-all" => self.select_all(),
            // The phone's way out of the clip tools: nothing selected, the
            // way a press on an empty lane leaves it.
            "deselect" => {
                self.flush_commit();
                self.selection.clear();
                self.transition_selected = None;
            }
            // ⇧⌫, from the key table; plain ⌫ comes in as its own callback.
            "ripple-delete" => self.ripple_delete_selected(),
            // Copy takes the whole selection; see `copy_clips`.
            "copy" => self.copy_clips(None),
            "duplicate" | "mute" => {
                if let Some(id) = self.sole_selection() {
                    self.clip_action(&id, action);
                }
            }
            // The tray's picture verbs. Mirror is the flip the H key does,
            // without the setting that gates the key: a button on the tray
            // is not a key someone might press by accident.
            "mirror" => self.toggle_flip_h(),
            "rotate" => self.rotate_selected(),
            "reverse" => {
                if let Some(id) = self.sole_selection() {
                    self.reverse_clip(&id);
                }
            }
            "freeze" => {
                self.menu_target = None;
                self.freeze_at_playhead();
            }
            // After the selected clip, on its lane, as the menu does; with
            // nothing selected, at the playhead on the rows copied from.
            "paste" => {
                let after = self.sole_selection();
                self.paste_clips(after.as_deref());
            }
            _ => {}
        }
    }

    /// One of the clip menu's verbs on clip `id`. The menu's rows and the
    /// keyboard's chords both land here, so a shortcut and the row that
    /// advertises it cannot disagree.
    pub fn clip_action(&mut self, id: &str, action: &str) {
        log::info!("clip menu: {action} on {id}");
        let Some(clip) = self.clip(id).cloned() else {
            return;
        };
        match action {
            "copy" => self.copy_clips(Some(id)),
            "duplicate" => self.duplicate_selected(),
            // Pasted after the clip that was right-clicked, on its lane.
            "paste" => self.paste_clips(Some(id)),
            "split" => {
                let at = self.playhead;
                self.flush_commit();
                self.selection = vec![id.to_owned()];
                self.split_at(at, true);
            }
            "freeze" => self.freeze_at_playhead(),
            // The menu opens the clip's Enhance in the inspector, where its
            // switch starts it and later turns it off and on again.
            "enhance" => {
                self.flush_commit();
                self.selection = vec![id.to_owned()];
                self.transition_selected = None;
                self.inspector_jump = (self.inspector_jump.0 + 1, "Video", "Enhance");
            }
            "export-audio" => self.export_clip_audio(id),
            "detach" => {
                self.apply(Command::DetachAudio {
                    clip_id: id.to_owned(),
                });
            }
            "reattach" => {
                self.apply(Command::ReattachAudio {
                    clip_id: id.to_owned(),
                });
            }
            "mute" => {
                let volume = if clip.volume <= 0.0 { 1.0 } else { 0.0 };
                self.apply(Command::UpdateClip {
                    clip_id: id.to_owned(),
                    patch: ClipPatch {
                        volume: Some(volume),
                        ..Default::default()
                    },
                });
            }
            "lock" => self.toggle_lock(&clip.track_id),
            // A clip that is part of the selection takes the selection with
            // it: Delete on one of five selected clips means the five.
            "delete" | "ripple-delete" => {
                let ripple = action == "ripple-delete" || self.prefs.magnetic;
                if self.selection.len() > 1 && self.selection.iter().any(|held| held == id) {
                    self.remove_selected(ripple);
                } else {
                    self.apply(Command::RemoveClips {
                        clip_ids: vec![id.to_owned()],
                        ripple,
                    });
                }
                self.menu_target = None;
            }
            // Q W I O X E R P [ ]: the Montage menu's keys.
            "delete-left" | "delete-right" | "mark-in" | "mark-out" | "clear-marks" | "extract"
            | "lift" | "play-add" | "play-prev" | "play-next" | "pointer-tool" | "back-five"
            | "forward-five" => self.montage_action(action),
            _ => {}
        }
    }

    /// Every clip on an unlocked lane.
    pub fn select_all(&mut self) {
        self.flush_commit();
        self.selection = self
            .timeline()
            .clips
            .iter()
            .filter(|clip| !self.locked(&clip.track_id))
            .map(|clip| clip.id.clone())
            .collect();
    }

    /// Tab `from` dropped at `slot`, a position counted over the strip as
    /// it stands; the command counts with the tab already removed.
    pub fn move_timeline(&mut self, from: i32, slot: i32) {
        let Some(from) = usize::try_from(from).ok() else {
            return;
        };
        let Some(timeline) = self.project().timelines.get(from) else {
            return;
        };
        let id = timeline.id.clone();
        let slot = slot.max(0) as usize;
        let index = if from < slot { slot - 1 } else { slot };
        if index == from {
            return;
        }
        self.apply(Command::MoveTimeline {
            timeline_id: id,
            index,
        });
    }
}

/// The clips a flattened edit makes audible, as playback takes them:
/// through the exporter's own cut into pieces, so a curve or a reverse
/// sounds in the window as it will in the file.
pub(crate) fn audio_specs(clips: &[concat_export::ExportClip]) -> Vec<ClipSpec> {
    clips
        .iter()
        .filter(|clip| {
            !clip.muted
                && (clip.kind == concat_export::ClipKind::Audio
                    || (clip.kind == concat_export::ClipKind::Video
                        && clip.has_audio.unwrap_or(true)))
        })
        // Through the exporter's own cut into pieces, so a curve or a
        // reverse sounds in the window as it will in the file.
        .flat_map(concat_export::audio_pieces)
        .map(|piece| ClipSpec {
            path: piece.path.to_string_lossy().into_owned(),
            audio_stream: piece.stream.map(|index| index as u32),
            start: piece.start,
            duration: piece.duration,
            source_start: piece.source_start,
            volume: piece.volume as f32,
            volume_curve: piece
                .volume_curve
                .keys()
                .iter()
                .map(|key| concat_host::playback::GainKey {
                    at: key.at,
                    gain: key.value,
                    ease: [key.ease.x1, key.ease.y1, key.ease.x2, key.ease.y2],
                })
                .collect(),
            fade_in: piece.fade_in,
            fade_out: piece.fade_out,
            speed: piece.speed,
            preserve_pitch: piece.preserve_pitch,
            chain: piece.filter_chain,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        Command, Footprint, Studio, adjust_key_names, chain_colours, chain_rows, custom_frame,
        custom_rate, fps_of, grading_rows, home_folder, key_commands, packed, place_in, shown,
        wheel_partners, write_keyable,
    };

    /// A pasted group keeps its shape from `base` on, its rows shifted
    /// and held inside the lanes it lands in, a locked row passed over
    /// for the nearest open one; with no lane at all nothing lands.
    #[test]
    fn a_paste_keeps_the_groups_shape_and_lands_on_rows_that_exist() {
        use concat_project::model::{Clip, ClipKind};
        let held = |row: usize, start: f64| super::Held {
            row,
            clip: Clip::blank("c", "t", ClipKind::Video, "c", start, 1.0),
        };
        let group = [held(1, 4.0), held(2, 5.5), held(0, 9.0)];
        let none = |_: usize| false;
        // At the playhead on the rows copied from.
        let plan = super::paste_plan(&group, 10.0, 0, 4, &none);
        assert_eq!(
            plan,
            vec![Some((1, 10.0)), Some((2, 11.5)), Some((0, 15.0))]
        );
        // After an anchor on row 3: the earliest lands there, the rest
        // keep their distance in rows, held inside the four lanes.
        let plan = super::paste_plan(&group, 20.0, 2, 4, &none);
        assert_eq!(
            plan,
            vec![Some((3, 20.0)), Some((3, 21.5)), Some((2, 25.0))]
        );
        // Another timeline with two lanes: rows past the end fold back.
        let plan = super::paste_plan(&group, 0.0, 0, 2, &none);
        assert_eq!(plan, vec![Some((1, 0.0)), Some((1, 1.5)), Some((0, 5.0))]);
        // A locked row is passed over for the nearest open one.
        let plan = super::paste_plan(&group, 0.0, 0, 4, &|row| row == 1);
        assert_eq!(plan, vec![Some((0, 0.0)), Some((2, 1.5)), Some((0, 5.0))]);
        // No lanes, nothing lands; every lane locked, nothing lands.
        assert!(
            super::paste_plan(&group, 0.0, 0, 0, &none)
                .iter()
                .all(Option::is_none)
        );
        assert!(
            super::paste_plan(&group, 0.0, 0, 2, &|_| true)
                .iter()
                .all(Option::is_none)
        );
    }

    /// A wheel is keyed as one knob: named by itself or by any of its own
    /// keys, it keys its puck and master; its master's key takes the puck's
    /// with it in the keyframes pane; any other knob keys alone.
    #[test]
    fn a_wheel_keys_as_one_knob() {
        let names = |key: &str| -> Vec<String> {
            adjust_key_names(key)
                .into_iter()
                .map(|(name, _)| name)
                .collect()
        };
        assert_eq!(names("lift"), ["lift.x", "lift.y", "lift.m"]);
        assert_eq!(names("gain.x"), ["gain.x", "gain.y", "gain.m"]);
        assert_eq!(names("exposure"), ["exposure"]);
        assert_eq!(wheel_partners("gamma.m"), ["gamma.x", "gamma.y"]);
        assert!(wheel_partners("gamma.x").is_empty());
        assert!(wheel_partners("exposure").is_empty());
    }

    /// The Adjust panel's wheels and curves come from its manifest: three
    /// wheels at rest and four straight curves on a clip that carries
    /// nothing, and a link's own values where it holds them.
    #[test]
    fn the_adjust_panel_draws_its_wheels_and_curves() {
        use concat_project::model::AppliedFilter;
        let package = concat_effects::Catalogue::builtin()
            .get("concat.adjust")
            .expect("a built-in");
        let (wheels, curves) = grading_rows(package, None, -1, Some(None));
        assert_eq!(
            wheels.iter().map(|w| w.key.as_str()).collect::<Vec<_>>(),
            ["lift", "gamma", "gain"]
        );
        assert!(
            wheels
                .iter()
                .all(|w| w.x == 0.0 && w.y == 0.0 && w.master == 0.0 && w.keyable)
        );
        assert_eq!(
            curves
                .iter()
                .map(|(c, _)| c.key.as_str())
                .collect::<Vec<_>>(),
            ["luma", "red", "green", "blue"]
        );
        assert!(curves.iter().all(|(_, points)| points.is_empty()));
        let mut link = AppliedFilter::new("concat.adjust");
        link.params.insert("gain.x".to_owned(), 0.25);
        link.params.insert("gain.m".to_owned(), -0.5);
        crate::grading::set_curve_point(&mut link.params, "red", -1, 0.5, 0.6);
        let (wheels, curves) = grading_rows(package, Some(&link), -1, None);
        assert_eq!((wheels[2].x, wheels[2].master), (0.25, -0.5));
        assert!(!wheels[2].keyable, "a stack's wheels take no keys");
        assert_eq!(curves[1].1.len(), 3, "the ends and the point put down");
    }

    /// A colour knob is a swatch in the stack, not a slider: broadcast
    /// green at the chroma key's default, and a colour picked is packed into
    /// the knob's number as the shader unpacks it, and read back the same.
    #[test]
    fn a_colour_knob_is_a_swatch() {
        use concat_project::model::AppliedFilter;
        let mut link = AppliedFilter::new("concat.chroma-key");
        let (_, knobs) = chain_rows(std::slice::from_ref(&link));
        assert!(knobs.iter().all(|knob| knob.key != "color"), "no slider");
        assert!(knobs.iter().any(|knob| knob.key == "similarity"));
        let colours = chain_colours(std::slice::from_ref(&link));
        assert_eq!(colours.len(), 1);
        assert_eq!(colours[0].key, "color");
        assert_eq!(
            colours[0].value,
            slint::Color::from_argb_u8(255, 0x00, 0xb1, 0x40)
        );
        let blue = slint::Color::from_argb_u8(255, 0x00, 0x47, 0xbb);
        link.params.insert("color".to_owned(), packed(blue));
        assert_eq!(link.params["color"], f64::from(0x0047_bbffu32));
        assert_eq!(chain_colours(&[link])[0].value, blue);
    }

    /// A typed frame is even on both sides and inside the limits.
    #[test]
    fn a_typed_frame_is_even_and_bounded() {
        assert_eq!(custom_frame(1920.0, 1080.0), (1920, 1080));
        assert_eq!(custom_frame(1081.0, 1079.0), (1082, 1080));
        assert_eq!(custom_frame(3.0, 99999.0), (16, 8192));
        assert_eq!(custom_frame(f32::NAN, -5.0), (16, 16));
    }

    /// A typed rate is the exact fraction: the NTSC ones as cameras record
    /// them, the rest to a thousandth in lowest terms, within the limits.
    #[test]
    fn a_typed_rate_is_an_exact_fraction() {
        assert_eq!(custom_rate(29.97), (30000, 1001));
        assert_eq!(custom_rate(23.976), (24000, 1001));
        assert_eq!(custom_rate(59.94), (60000, 1001));
        assert_eq!(custom_rate(30.0), (30, 1));
        assert_eq!(custom_rate(12.5), (25, 2));
        assert_eq!(custom_rate(0.2), (1, 1));
        assert_eq!(custom_rate(1000.0), (240, 1));
        assert!((fps_of(30000, 1001) - 29.97).abs() < 0.001);
    }

    const FRAME: (u32, u32) = (1920, 1080);

    /// A suggested folder sits under the home directory, with the
    /// platform's separators throughout, and is never empty on a machine
    /// that has a home: the launch sheet's Location starts from it.
    #[test]
    fn home_folder_is_under_home_with_native_separators() {
        let home = std::env::home_dir().expect("a home directory");
        let folder = std::path::PathBuf::from(home_folder("Desktop/Concat"));
        assert_eq!(folder, home.join("Desktop").join("Concat"));
        #[cfg(windows)]
        assert!(!folder.to_string_lossy().contains('/'));
    }

    /// A quarter turn swaps the bounds' pixel extents, which in fractions
    /// of a 16:9 frame is not a swap of the numbers.
    #[test]
    fn half_bounds_follow_the_turn() {
        let flat = Footprint {
            cx: 0.5,
            cy: 0.5,
            w: 0.5,
            h: 0.5,
            rotation: 0.0,
        };
        let (hw, hh) = flat.half_bounds(FRAME);
        assert!((hw - 0.25).abs() < 1e-9 && (hh - 0.25).abs() < 1e-9);
        let turned = Footprint {
            rotation: 90.0,
            ..flat
        };
        let (hw, hh) = turned.half_bounds(FRAME);
        // 540px tall becomes 540px wide: 270px each side of 1920.
        assert!((hw - 270.0 / 1920.0).abs() < 1e-9);
        assert!((hh - 480.0 / 1080.0).abs() < 1e-9);
    }

    /// The nearest pair inside the pull wins, and nothing outside it pulls.
    #[test]
    fn snap_picks_the_nearest_target_inside_the_pull() {
        // Centre 0.5 sits just off the frame's centre; the left edge at
        // 0.2 is nearer to another picture's edge at 0.21.
        let features = [0.2, 0.5, 0.8];
        let hit = Studio::stage_snap(features, &[0.0, 0.505, 1.0, 0.21], 0.02).unwrap();
        assert!((hit.0 - 0.005).abs() < 1e-9 && (hit.1 - 0.505).abs() < 1e-9);
        assert!(Studio::stage_snap(features, &[0.3, 0.6], 0.02).is_none());
    }

    /// A fitted 16:9 picture at rest covers the frame exactly.
    #[test]
    fn footprint_contains_an_unturned_box() {
        let box_ = Footprint {
            cx: 0.5,
            cy: 0.5,
            w: 0.5,
            h: 0.5,
            rotation: 0.0,
        };
        assert!(box_.contains(0.5, 0.5, FRAME));
        assert!(box_.contains(0.26, 0.26, FRAME));
        assert!(!box_.contains(0.24, 0.5, FRAME));
        assert!(!box_.contains(0.5, 0.76, FRAME));
    }

    /// Turned a quarter, a wide box stands tall: a point that was inside
    /// along the width is outside, and one above the old top edge is inside.
    /// The test is in pixels, which is where the turn happens — a fraction
    /// across is not the same distance as a fraction down.
    #[test]
    fn footprint_turns_in_pixels() {
        let box_ = Footprint {
            cx: 0.5,
            cy: 0.5,
            w: 0.5,
            h: 0.2,
            rotation: 90.0,
        };
        // Half the width is 480px, half the height 108px. After the turn
        // the box reaches 480px up and down and 108px either side.
        assert!(box_.contains(0.5, 0.5 + 400.0 / 1080.0, FRAME));
        assert!(!box_.contains(0.5, 0.5 + 500.0 / 1080.0, FRAME));
        assert!(box_.contains(0.5 + 100.0 / 1920.0, 0.5, FRAME));
        assert!(!box_.contains(0.5 + 120.0 / 1920.0, 0.5, FRAME));
    }

    /// The turn is clockwise, as the compositor's is. The unturned top-right
    /// corner sits at (480, -270) from the centre; turned thirty degrees
    /// clockwise in y-down pixels it lands at about (551, +6) - it has swung
    /// *down* past the centre line. A point just inside that corner is in
    /// the box, and its mirror above the line is not; turned the other way
    /// both answers would flip.
    #[test]
    fn footprint_turns_clockwise() {
        let box_ = Footprint {
            cx: 0.5,
            cy: 0.5,
            w: 0.5,
            h: 0.5,
            rotation: 30.0,
        };
        let x = 0.5 + 540.0 / 1920.0;
        assert!(box_.contains(x, 0.5 + 6.0 / 1080.0, FRAME));
        assert!(!box_.contains(x, 0.5 - 6.0 / 1080.0, FRAME));
    }

    /// A knob over a property with no keys writes the constant; over one
    /// that rides, it writes the key at the playhead and leaves the
    /// constant alone. The echo's keys then become one command each.
    /// A batch's position and turn move by the knob's change and its scale
    /// by the knob's factor; the first clip, the knob's reference, lands on
    /// the knob's value, and a field that is not a placement is set outright.
    #[test]
    fn a_batch_keeps_its_shape_about_the_first_clip() {
        use super::{ClipField, batch_value, relative_property};
        // The first clip was at 0.1 and the knob now reads 0.3: everyone
        // moves right by 0.2.
        assert_eq!(batch_value(ClipField::OffsetX, 0.3, 0.1, 0.1), 0.3);
        assert!((batch_value(ClipField::OffsetX, 0.3, 0.1, -0.5) - -0.3).abs() < 1e-12);
        assert_eq!(batch_value(ClipField::Rotation, 90.0, 0.0, 45.0), 135.0);
        // Scale is a factor: the first doubled, so the second doubles.
        assert_eq!(batch_value(ClipField::Scale, 2.0, 1.0, 0.5), 1.0);
        // A size is a size for every title in the batch.
        assert_eq!(batch_value(ClipField::FontSize, 0.08, 0.05, 0.02), 0.08);
        assert_eq!(relative_property(ClipField::FontSize), None);
    }

    #[test]
    fn a_keyed_property_is_written_as_a_key_and_committed_as_commands() {
        use concat_project::model::{Clip, ClipKind, KeyEase, KeyProperty};
        let before = Clip::blank("c1", "t1", ClipKind::Video, "clip", 0.0, 10.0);
        let mut after = before.clone();
        write_keyable(&mut after, KeyProperty::Scale, 2.0, 0.5);
        assert_eq!(after.scale, 2.0, "no keys: the constant");
        assert!(key_commands("c1", &before, &after).is_empty());

        after.set_key(KeyProperty::Scale, 0.25, 1.0, KeyEase::LINEAR);
        write_keyable(&mut after, KeyProperty::Scale, 3.0, 0.75);
        assert_eq!(after.scale, 2.0, "keyed: the constant is left alone");
        assert_eq!(after.value_at(KeyProperty::Scale, 0.75), 3.0);
        assert_eq!(shown(&after, KeyProperty::Scale, 0.75), 3.0);
        let commands = key_commands("c1", &before, &after);
        assert_eq!(commands.len(), 2, "one per key: {commands:?}");
        assert!(commands.iter().all(|command| matches!(
            command,
            Command::SetClipKey {
                property: KeyProperty::Scale,
                ..
            }
        )));

        let mut gone = after.clone();
        gone.clear_key(KeyProperty::Scale, 0.25);
        let commands = key_commands("c1", &after, &gone);
        assert!(
            matches!(commands.as_slice(), [Command::ClearClipKey { at, .. }] if (*at - 0.25).abs() < 1e-9),
            "{commands:?}"
        );
        assert_eq!(place_in(&after, 2.5), 0.25);
        assert_eq!(place_in(&after, 12.0), 1.0, "held to the end");
    }
}

/// The coach's library: game-model profiles, reference examples, the
/// models learned from them, and the coach's own actions.
#[derive(Default)]
pub struct EvalLibrary {
    /// The game-model profiles.
    pub profiles: Vec<concat_vision::football::eval::Profile>,
    /// The examples marked good or bad.
    pub references: Vec<concat_vision::football::eval::Reference>,
    /// The models learned from the examples, one per action.
    pub models: Vec<concat_vision::football::learn::Model>,
    /// The coach's own actions.
    pub actions: Vec<concat_vision::football::learn::ActionDef>,
}

/// One window judged: its action's key, what was measured, and the verdict.
pub struct Judged {
    /// The action's key.
    pub key: String,
    /// What was measured.
    pub measures: concat_vision::football::eval::Measures,
    /// The verdict and its reasons.
    pub judgement: concat_vision::football::eval::Judgement,
}

/// The match a project's name reads as, for the broadcast scoreboard: the
/// first `<n>-<m>` standing on its own (an en dash too), the sides before
/// and after it, each cut to the first three letters of its first word in
/// capitals. "Rayados Sub-17 2-3 Necaxa" reads as ("RAY", "2 – 3", "NEC");
/// a name with no such score, or nothing on one side of it, reads as none.
fn scoreboard(name: &str) -> Option<(String, String, String)> {
    let words: Vec<&str> = name.split_whitespace().collect();
    let goals = |text: &str| {
        !text.is_empty() && text.len() <= 2 && text.chars().all(|c| c.is_ascii_digit())
    };
    let (at, home_goals, away_goals) = words.iter().enumerate().find_map(|(i, word)| {
        let (a, b) = word.split_once('-').or_else(|| word.split_once('–'))?;
        (goals(a) && goals(b)).then(|| (i, a.to_owned(), b.to_owned()))
    })?;
    let short = |side: &[&str]| -> Option<String> {
        let first = side.first()?;
        let short: String = first
            .chars()
            .filter(|c| c.is_alphanumeric())
            .take(3)
            .collect::<String>()
            .to_uppercase();
        (!short.is_empty()).then_some(short)
    };
    Some((
        short(&words[..at])?,
        format!("{home_goals} – {away_goals}"),
        short(&words[at + 1..])?,
    ))
}

#[cfg(test)]
mod scoreboard_tests {
    use super::scoreboard;

    #[test]
    fn a_match_name_reads_as_a_scoreboard() {
        assert_eq!(
            scoreboard("Rayados Sub-17 2-3 Necaxa"),
            Some(("RAY".into(), "2 – 3".into(), "NEC".into()))
        );
        assert_eq!(
            scoreboard("Atlas 1–0 Chivas"),
            Some(("ATL".into(), "1 – 0".into(), "CHI".into()))
        );
    }

    #[test]
    fn other_names_read_as_none() {
        assert_eq!(scoreboard("Biblioteca Atlas"), None);
        assert_eq!(scoreboard("2-3 Necaxa"), None);
        assert_eq!(scoreboard("Sesión 2026-10-21"), None);
    }
}

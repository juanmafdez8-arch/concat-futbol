// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! The football analysis on real footage, end to end: detect and track,
//! read one player's silhouettes, then draw every mark on a few frames and
//! write them out to look at.
//!
//! Needs the detectors in `<app data>/football-models/` and a clip, so it
//! is ignored by default:
//!
//! ```text
//! CONCAT_FOOTBALL_VIDEO=clip.mp4 CONCAT_FOOTBALL_OUT=out \
//!     cargo test -p concat-host --test football_real -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use concat_core::frame::{BYTES_PER_PIXEL, Frame};
use concat_core::time::Rational;
use concat_host::cutout::Progress;
use concat_host::{AppDirs, Brushes, FootballRequest, Footballs};
use concat_media::{DecodeOptions, Decoder, FrameSource};
use concat_vision::Mapping;
use concat_vision::football::{self, BALL, Kind, TrackSet, marks};

fn write_ppm(frame: &Frame, path: &std::path::Path) {
    let mut bytes = format!("P6\n{} {}\n255\n", frame.width(), frame.height()).into_bytes();
    for pixel in frame.pixels().chunks_exact(BYTES_PER_PIXEL) {
        // Transparent shows as magenta, so an isolated player can be seen.
        let a = f32::from(pixel[3]) / 255.0;
        for (c, back) in [255.0, 0.0, 255.0].iter().enumerate() {
            bytes.push((f32::from(pixel[c]) * a + back * (1.0 - a)) as u8);
        }
    }
    std::fs::write(path, bytes).expect("write frame");
}

#[test]
#[ignore]
fn football_on_real_footage() {
    let video = std::env::var("CONCAT_FOOTBALL_VIDEO").expect("CONCAT_FOOTBALL_VIDEO");
    let out = PathBuf::from(std::env::var("CONCAT_FOOTBALL_OUT").unwrap_or_else(|_| "out".into()));
    std::fs::create_dir_all(&out).unwrap();
    let dirs = AppDirs::locate().unwrap();
    let footballs = Footballs::new(&dirs.data, std::sync::Arc::new(Brushes::new(&dirs.data)));
    assert!(footballs.installed(), "no football model in {}", dirs.data.display());

    let probe = concat_media::probe(&video).expect("probe");
    let stream = probe.require_video().expect("a video stream");
    let size = (stream.width, stream.height);
    let duration = probe.duration.map(|d| d.as_f64()).unwrap_or(0.0);
    let request = FootballRequest {
        project: out.clone(),
        media_path: video.clone(),
        media_size: size,
        ranges: vec![(0.0, duration)],
    };

    let started = std::time::Instant::now();
    let analysed = footballs
        .analyse(&request, None, &mut |_: Progress| {}, &mut || {})
        .expect("analyse");
    let set = TrackSet::load(&request.dir()).expect("tracks written");
    let tracks = set.tracks();
    let people = tracks.iter().filter(|t| t.1 != Kind::Ball).count();
    let per_frame = set.samples.iter().map(|s| s.boxes.len()).sum::<usize>() as f32
        / set.samples.len().max(1) as f32;
    let ball_frames = set
        .samples
        .iter()
        .filter(|s| s.boxes.iter().any(|b| b.id == BALL))
        .count();
    println!(
        "analysed {analysed} instants in {:.1}s: {people} people tracks, {per_frame:.1} boxes/instant, ball in {ball_frames}",
        started.elapsed().as_secs_f32()
    );
    let lengths: Vec<usize> = tracks.iter().take(12).map(|t| t.2).collect();
    println!("longest tracks (instants): {lengths:?}");
    assert!(people > 0, "nobody found");

    // The player seen longest is the one marked.
    let (chosen, _, _) = *tracks
        .iter()
        .find(|t| t.1 != Kind::Ball)
        .expect("a player");
    let started = std::time::Instant::now();
    let written = footballs
        .silhouettes(&request, chosen, &mut |_: Progress| {})
        .expect("silhouettes");
    println!(
        "silhouettes of #{chosen}: {written} in {:.1}s",
        started.elapsed().as_secs_f32()
    );
    let store = football::silhouettes(&request.dir(), chosen);

    // A frame a second: the original and every mark on it.
    for second in 0..(duration as u32).min(5) {
        let at = f64::from(second) + 0.5;
        let options = DecodeOptions::default()
            .starting_at(Rational::new((at * 1000.0) as i64, 1000))
            .limited_to(1);
        let mut decoder = Decoder::open(&video, &options).unwrap();
        let Some(frame) = decoder.next_frame().unwrap() else {
            continue;
        };
        let Some(tracked) = set.track_at(chosen, at) else {
            println!("t={at}: #{chosen} not in view");
            continue;
        };
        let mask = store.mask_at(at);
        let shift = store
            .nearest_millis(at)
            .and_then(|ms| set.track_at(chosen, ms as f64 / 1000.0))
            .map(|then| {
                let (a, b) = (tracked.centre(), then.centre());
                (a.0 - b.0, a.1 - b.1)
            })
            .unwrap_or((0.0, 0.0));
        let target = marks::Target {
            bbox: tracked.bbox,
            silhouette: mask.as_deref(),
            shift,
        };
        let mapping = Mapping::IDENTITY;
        let boxes: Vec<_> = set
            .at(at)
            .iter()
            .map(|b| (b.bbox, if b.id == chosen { [203, 245, 63] } else { [255, 255, 255] }, b.id == chosen))
            .collect();
        let mut picked = frame.clone();
        marks::outline(&mut picked, &boxes, &mapping);
        write_ppm(&picked, &out.join(format!("t{second}_0_pick.ppm")));
        let mut shaded = frame.clone();
        marks::shade(&mut shaded, &target, &mapping, [203, 245, 63], 0.8);
        marks::ring(&mut shaded, &target, &mapping, set.aspect, [203, 245, 63], 0.9, marks::Shape::DEFAULT);
        marks::arrow(&mut shaded, &target, &mapping, set.aspect, [203, 245, 63], 0.9, marks::Shape::DEFAULT);
        write_ppm(&shaded, &out.join(format!("t{second}_1_shade_ring.ppm")));
        let mut lit = frame.clone();
        marks::spotlight(&mut lit, &[(target, marks::Shape::DEFAULT)], &mapping, set.aspect, 0.8);
        write_ppm(&lit, &out.join(format!("t{second}_2_spotlight.ppm")));
        let mut erased = frame.clone();
        marks::erase(&mut erased, &target, &mapping, 1.3);
        write_ppm(&erased, &out.join(format!("t{second}_3_erase.ppm")));
        let mut alone = frame.clone();
        marks::isolate(&mut alone, &target, &mapping);
        write_ppm(&alone, &out.join(format!("t{second}_4_isolate.ppm")));
        println!(
            "t={at}: #{chosen} at {:?}, silhouette {}",
            tracked.bbox,
            if mask.is_some() { "yes" } else { "no (box)" }
        );
    }
}

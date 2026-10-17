// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! Where the football analysis spends its time: decoding, each detector,
//! the tracker and the brush model, measured apart on real footage.
//!
//! ```text
//! CONCAT_FOOTBALL_VIDEO=clip.mp4 \
//!     cargo test --release -p concat-host --test football_bench -- --ignored --nocapture
//! ```

use std::time::Instant;

use concat_core::time::{FrameRate, Rational};
use concat_host::AppDirs;
use concat_media::{DecodeOptions, Decoder, FrameSource};
use concat_vision::football::detect::{Detector, silhouette};
use concat_vision::football::track;
use concat_vision::{Brush, ModelId, models};

const SIZE: usize = 1280;

fn ms(since: Instant, count: usize) -> String {
    format!(
        "{:.1} ms/frame",
        since.elapsed().as_secs_f64() * 1000.0 / count.max(1) as f64
    )
}

#[test]
#[ignore]
fn football_bench() {
    let video = std::env::var("CONCAT_FOOTBALL_VIDEO").expect("CONCAT_FOOTBALL_VIDEO");
    let dirs = AppDirs::locate().unwrap();
    let folder = dirs.data.join("football-models");
    let probe = concat_media::probe(&video).unwrap();
    let stream = probe.require_video().unwrap();
    let (width, height) = (stream.width, stream.height);

    for rate in [10i64, 5] {
        let started = Instant::now();
        let options = DecodeOptions::default()
            .scaled_to(SIZE as u32, (SIZE as u32 * height / width).max(1))
            .at_rate(FrameRate::new(Rational::new(rate, 1)));
        let mut decoder = Decoder::open(&video, &options).unwrap();
        let mut frames = Vec::new();
        while let Some(frame) = decoder.next_frame().unwrap() {
            frames.push(frame);
        }
        println!("decode at {rate}/s: {} frames, {}", frames.len(), ms(started, frames.len()));
    }

    let options = DecodeOptions::default()
        .scaled_to(SIZE as u32, (SIZE as u32 * height / width).max(1))
        .at_rate(FrameRate::new(Rational::new(10, 1)));
    let mut decoder = Decoder::open(&video, &options).unwrap();
    let mut frames = Vec::new();
    while let Some(frame) = decoder.next_frame().unwrap() {
        frames.push(frame);
    }

    for (name, file) in [("football", "football.onnx"), ("general", "general.onnx")] {
        let path = folder.join(file);
        if !path.is_file() {
            println!("{name}: not installed");
            continue;
        }
        let started = Instant::now();
        let detector = Detector::load(&path, SIZE).unwrap();
        println!("{name}: loaded in {:.0} ms", started.elapsed().as_secs_f64() * 1000.0);
        // The first run warms the provider up; it is not the steady state.
        let started = Instant::now();
        detector.detect(&frames[0]).unwrap();
        println!("{name}: first frame {:.0} ms", started.elapsed().as_secs_f64() * 1000.0);
        let started = Instant::now();
        let mut found = 0;
        for frame in &frames {
            found += detector.detect(frame).unwrap().len();
        }
        println!("{name}: {} ({found} boxes)", ms(started, frames.len()));
    }

    let detector = Detector::load(&folder.join("football.onnx"), SIZE).unwrap();
    let detections: Vec<_> = frames
        .iter()
        .enumerate()
        .map(|(i, frame)| (i as u64 * 100, detector.detect(frame).unwrap()))
        .collect();
    let started = Instant::now();
    let samples = track::link(&detections, 100);
    println!("tracking: {}", ms(started, samples.len()));
    // The tracker's quality at ten instants a second and at five: how many
    // identities, and how long the longest dozen last, in seconds.
    let quality = |samples: &[concat_vision::football::Sample], step: f64| {
        let mut seen: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for sample in samples {
            for tracked in &sample.boxes {
                *seen.entry(tracked.id).or_insert(0) += 1;
            }
        }
        let mut lengths: Vec<f64> = seen.values().map(|n| *n as f64 * step).collect();
        lengths.sort_by(|a, b| b.total_cmp(a));
        lengths.truncate(12);
        (seen.len(), lengths)
    };
    let (ids, longest) = quality(&samples, 0.1);
    println!("tracks at 10/s: {ids} identities, longest {longest:?}");
    let halved: Vec<_> = detections.iter().step_by(2).cloned().collect();
    let samples5 = track::link(&halved, 200);
    let (ids, longest) = quality(&samples5, 0.2);
    println!("tracks at 5/s: {ids} identities, longest {longest:?}");

    if models::installed(&dirs.data, ModelId::BrushEncoder) {
        let brush = Brush::load(
            &models::model_file(&dirs.data, ModelId::BrushEncoder),
            &models::model_file(&dirs.data, ModelId::BrushDecoder),
        )
        .unwrap();
        let tracked = samples
            .iter()
            .flat_map(|s| s.boxes.iter().copied())
            .find(|b| b.kind != concat_vision::football::Kind::Ball)
            .expect("a player");
        silhouette(&brush, &frames[0], tracked.bbox, 1280).unwrap();
        let started = Instant::now();
        let count = 10.min(frames.len());
        for frame in frames.iter().take(count) {
            silhouette(&brush, frame, tracked.bbox, 1280).unwrap();
        }
        println!("silhouette: {}", ms(started, count));
    } else {
        println!("silhouette: brush model not installed");
    }
}

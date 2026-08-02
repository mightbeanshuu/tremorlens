---
title: TremorLens · SPANDAN Engine
emoji: 🌉
colorFrom: gray
colorTo: orange
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# TremorLens — Every Camera a Structural Health Sensor

Upload a video of a structure (with high-contrast speckle targets taped on) and
the full SPANDAN engine runs server-side: sub-pixel phase-correlation tracking,
modal fingerprint (f₁, damping), bootstrap confidence intervals — and with a
second "current state" video, damage verdict, novelty score, and per-region
transmissibility localization. Optional phyphox accelerometer CSV gives the
camera-vs-contact-sensor agreement overlay.

Live phone-sensor demo: https://tremorlens-live.vercel.app ·
Source: https://github.com/mightbeanshuu/tremorlens

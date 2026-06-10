<p align="center">
  <img src="assets/LogoAmora.png" width="120" alt="AMORA Logo"/>
</p>

# AMORA

AI-powered pregnancy fitness mobile app that detects movement in real-time via smartphone camera and provides instant correction if a movement is deemed unsafe.

**Platform:** iOS & Android  
**Category:** Wellness App (not a medical device)  
**Target Users:** Pregnant women and women across the reproductive cycle, ages 25–40, urban middle class Indonesia

---

## Current Scope: Squat Analysis — Trimester 1

### Architecture Overview
Safety    → Rule-based (immutable, clinical basis)
Quality   → ML Model (1D-CNN, trained from rule-based)
Feedback  → LLM (constrained, post-rep only)

### Clinical Basis

Squat thresholds for Trimester 1 are based on:
- ACOG (American College of Obstetricians and Gynecologists)
- Mayo Clinic pregnancy exercise guidelines
- Straub & Powers 2024

> AMORA is a wellness app, not a medical device. Always consult an OB-GYN before starting any exercise program during pregnancy.
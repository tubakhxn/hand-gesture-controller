Hand HUD — palm-anchored kinematic dashboard

This project captures webcam input, detects a hand using MediaPipe, and draws a white, mechanical-style kinematic HUD over the hand (palm-centered radial UI, finger bones, rotation readout, small 3D cube and grid) similar to the reference images.

## Files

- `requirements.txt` — Python dependencies
- `main.py` — application entrypoint and webcam loop
- `hand_overlay.py` — functions for drawing the HUD and computing kinematics
- `utils.py` — small helpers (smoothing, geometry)

## How to run (Windows PowerShell)

1. Create and activate a virtual environment (optional but recommended):

   python -m venv .venv; .\.venv\Scripts\Activate.ps1

2. Install dependencies:

   pip install -r requirements.txt

3. Run:

   python main.py

## Notes

- Works with a single hand in frame. For multi-hand, toggle a flag in `main.py`.
- Tweak smoothing and drawing constants in `hand_overlay.py`.
- If you want recording/output images, I can add that next.

## Copyright & Attribution

This repository is published publicly by the original creator: **tubakhxn**.

You are welcome to fork, clone, and contribute to this project. If you reuse substantial parts of the code or publish derivative works, please give clear credit to the original author. A suggested credit line is:

> Based on tubakhxn/hand-overlay — https://github.com/tubakhxn/hand-overlay

Please keep a copy of this README and the `LICENSE` file in redistributed or forked versions so attribution remains visible.

## License

This project is released under the MIT License — see the `LICENSE` file included in this repository. The MIT License permits reuse, modification, and redistribution; please preserve the copyright notice and give credit to the original author when distributing or republishing the work.

If you'd prefer a license that explicitly requires attribution (for example, Creative Commons Attribution 4.0), let me know and I can switch the license file.

## Contributing

Contributions are welcome. To contribute:

1. Open an issue to discuss proposed changes or file a bug report.
2. Create a branch for your changes and open a pull request with a clear description.
3. Keep changes focused and include tests or usage notes when appropriate.

By submitting a pull request you agree that your contribution will be made under the repository's license (MIT by default).

## Contact

If you want to reach out, open an issue or visit the author's GitHub profile: https://github.com/tubakhxn

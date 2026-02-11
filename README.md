[![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)](https://www.python.org/downloads/release/python-270/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<p align="center">
  <a href="https://www.mdpi.com/1424-8220/20/5/1497">
    <img src="https://github.com/user-attachments/assets/b724f8ec-949a-4e42-9ab8-dc3ccf876516" alt="ArUco 404" width="150"><br>
    <b><span style="font-size: 28px;">
      ArUco Inpainting
    </span></b>
  </a>
</p>

| Before | After |
|---|---|
| ![Before – with ArUco markers](https://github.com/user-attachments/assets/20174f80-efe2-4e7c-9089-df65bde5cfae) | ![After – clean image](https://github.com/user-attachments/assets/378a5978-e586-468c-8727-69a87b61c85b) |

Removes visible ArUco markers from RGB images while preserving their geometric role in multi-view 3D reconstruction.
It detects and projects marker geometry into each camera view to perform accurate inpainting, then reprojects the cleaned imagery onto point clouds to generate marker-free colored reconstructions.

---

## Citation

If you use **Meshtrics** in your work, please cite:

```bibtex
@Article{10.3390/s20051497,
AUTHOR = {Madeira, Tiago and Oliveira, Miguel and Dias, Paulo},
TITLE = {Enhancement of RGB-D Image Alignment Using Fiducial Markers},
JOURNAL = {Sensors},
VOLUME = {20},
YEAR = {2020},
NUMBER = {5},
ARTICLE-NUMBER = {1497},
DOI = {10.3390/s20051497}
}
```
---

## License

Distributed under the **GPL-3.0 License**. See [`LICENSE`](LICENSE) for more information.

# POVray Visualization Example

A simple example of rendering pyelastica using POVray.
The code [render](continuum_snake_render.py) generates POVray script (.pov) and image file (.png) to render POVray animation.

### Bash Script to Run
``` bash
python continuum_snake.py         # Creates continuum_snake.dat file
python continuum_snake_render.py  # Creates pov_snake_diag.mp4 and pov_snake_top.mp4 file (3-5 minutes)
```

### Dependency
- povray
- ffmpeg

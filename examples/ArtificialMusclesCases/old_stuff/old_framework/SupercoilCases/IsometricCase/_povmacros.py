""" POVray macros for pyelastica

This module includes utility methods to support POVray rendering.

"""

import subprocess
from collections import defaultdict


def pyelastica_rod(
    x,
    r,
    color="rgb<0.45,0.39,1>",
    transmit=0.0,
    interpolation="linear_spline",
    deform=None,
    tab="    ",
):
    """pyelastica_rod POVray script generator

    Generates povray sphere_sweep object in string.
    The rod is given with the element radius (r) and joint positions (x)

    Parameters
    ----------
    x : numpy array
        Position vector
        Expected shape: [num_time_step, 3, num_element]
    r : numpy array
        Radius vector
        Expected shape: [num_time_step, num_element]
    color : str
        Color of the rod (default: Purple <0.45,0.39,1>)
    transmit : float
        Transparency (0.0 to 1.0).
    interpolation : str
        Interpolation method for sphere_sweep
        Supporting type: 'linear_spline', 'b_spline', 'cubic_spline'
        (default: linear_spline)
    deform : str
        Additional object deformation
        Example: "scale<4,4,4> rotate<0,90,90> translate<2,0,4>"

    Returns
    -------
    cmd : string
        Povray script
    """

    assert interpolation in ["linear_spline", "b_spline", "cubic_spline"]
    tab = "    "

    # Parameters
    num_element = r.shape[0]

    lines = []
    lines.append("sphere_sweep {")
    lines.append(tab + f"{interpolation} {num_element}")
    for i in range(num_element):
        lines.append(tab + f",<{x[0,i]},{x[1,i]},{x[2,i]}>,{r[i]}")
    lines.append(tab + "texture{")
    lines.append(tab + tab + "pigment{ color %s transmit %f }" % (color, transmit))
    lines.append(tab + tab + "finish{ phong 1 }")
    lines.append(tab + "}")
    if deform is not None:
        lines.append(tab + deform)
    lines.append(tab + "}\n")

    cmd = "\n".join(lines)
    return cmd


def render(
    filename, width, height, antialias="on", quality=11, display="Off", pov_thread=4
):
    """Rendering frame

    Generate the povray script file '.pov' and image file '.png'
    The directory must be made before calling this method.

    Parameters
    ----------
    filename : str
        POV filename (without extension)
    width : int
        The width of the output image.
    height : int
        The height of the output image.
    antialias : str ['on', 'off']
        Turns anti-aliasing on/off [default='on']
    quality : int
        Image output quality. [default=11]
    display : str
        Turns display option on/off during POVray rendering. [default='off']
    pov_thread : int
        Number of thread per povray process. [default=4]
        Acceptable range is (4,512).
        Refer 'Symmetric Multiprocessing (SMP)' for further details
        https://www.povray.org/documentation/3.7.0/r3_2.html#r3_2_8_1

    Raises
    ------
    IOError
        If the povray run causes unexpected error, such as parsing error,
        this method will raise IOerror.

    """

    # Define script path and image path
    script_file = filename + ".pov"
    image_file = filename + ".png"

    # Run Povray as subprocess
    cmds = [
        "povray",
        "+I" + script_file,
        "+O" + image_file,
        f"-H{height}",
        f"-W{width}",
        f"Work_Threads={pov_thread}",
        f"Antialias={antialias}",
        f"Quality={quality}",
        f"Display={display}",
    ]
    process = subprocess.Popen(
        cmds, stderr=subprocess.PIPE, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    _, stderr = process.communicate()

    # Check execution error
    if process.returncode:
        print(type(stderr), stderr)
        raise IOError(
            "POVRay rendering failed with the following error: "
            + stderr.decode("ascii")
        )


class Stages:
    """Stage definition

    Collection of the camera and light sources.
    Each camera added to the stage represent distinct viewpoints to render.
    Lights can be assigned to multiple cameras.
    The povray script can be generated for each viewpoints created using 'generate_scripts.'

    (TODO) Implement transform camera for dynamic camera moves

    Attributes
    ----------
    pre_scripts : str
        Prepending script for all viewpoints
    post_scripts : str
        Appending script for all viewpoints
    cameras : list
        List of camera setup
    lights : list
        List of lightings
    _light_assign : dictionary[list]
        Dictionary that pairs lighting to camera.
        Example) _light_assign[2] is the list of light sources
            assigned to the cameras[2]

    Methods
    -------
    add_camera : Add new camera (viewpoint) to the stage.
    add_light : Add new light source to the stage for a assigned camera.
    generate_scripts : Generate list of povray script for each camera.

    Class Objects
    -------------
    StageObject
    Camera
    Light

    Properties
    ----------
    len : number of camera
        The number of viewpoints
    """

    def __init__(self, pre_scripts="", post_scripts=""):
        self.pre_scripts = pre_scripts
        self.post_scripts = post_scripts
        self.cameras = []
        self.lights = []
        self._light_assign = defaultdict(list)

    def add_camera(self, name, **kwargs):
        """Add camera (viewpoint)"""
        self.cameras.append(self.Camera(name=name, **kwargs))

    def add_light(self, camera_id=-1, **kwargs):
        """Add lighting and assign to camera
        Parameters
        ----------
        camera_id : int or list
            Assigned camera. [default=-1]
            If a list of camera_id is given, light is assigned for listed camera.
            If camera_id==-1, the lighting is assigned for all camera.
        """
        light_id = len(self.lights)
        self.lights.append(self.Light(**kwargs))
        if isinstance(camera_id, list) or isinstance(camera_id, tuple):
            camera_id = list(set(camera_id))
            for idx in camera_id:
                self._light_assign[idx].append(light_id)
        elif isinstance(camera_id, int):
            self._light_assign[camera_id].append(light_id)
        else:
            raise NotImplementedError("camera_id can only be a list or int")

    def generate_scripts(self):
        """Generate pov-ray script for all camera setup
        Returns
        -------
        scripts : list
            Return list of pov-scripts (string) that includes camera and assigned lightings.
        """
        scripts = {}
        for idx, camera in enumerate(self.cameras):
            light_ids = self._light_assign[idx] + self._light_assign[-1]
            cmds = []
            cmds.append(self.pre_scripts)
            cmds.append(str(camera))  # Script camera
            for light_id in light_ids:  # Script Lightings
                cmds.append(str(self.lights[light_id]))
            cmds.append(self.post_scripts)
            scripts[camera.name] = "\n".join(cmds)
        return scripts

    def transform_camera(self, dx, R, camera_id):
        # (TODO) translate or rotate the assigned camera
        raise NotImplementedError

    def __len_(self):
        return len(self.cameras)

    # Stage Objects: Camera, Light
    class StageObject:
        """Template for stage objects

        Objects (camera and light) is defined as an object in order to
        manipulate (translate or rotate) them during the rendering.

        Attributes
        ----------
        str : str
            String representation of object.
            The placeholder exist to avoid rescripting.

        Methods
        -------
        _color2str : str
            Change triplet tuple (or list) of color into rgb string.
        _position2str : str
            Change triplet tuple (or list) of position vector into string.
        """

        def __init__(self):
            self.str = ""
            self.update_script()

        def update_script(self):
            raise NotImplementedError

        def __str__(self):
            return self.str

        def _color2str(self, color):
            if isinstance(color, str):
                return color
            elif isinstance(color, list) and len(color) == 3:
                # RGB
                return "rgb<{},{},{}>".format(*color)
            else:
                raise NotImplementedError(
                    "Only string-type color or RGB input is implemented"
                )

        def _position2str(self, position):
            assert len(position) == 3
            return "<{},{},{}>".format(*position)

    class Camera(StageObject):
        """Camera object

        http://www.povray.org/documentation/view/3.7.0/246/

        Attributes
        ----------
        location : list or tuple
            Position vector of camera location. (length=3)
        angle : int
            Camera angle
        look_at : list or tuple
            Position vector of the location where camera points to (length=3)
        name : str
            Name of the view-point.
        sky : list or tuple
            Tilt of the camera (length=3) [default=[0,1,0]]
        """

        def __init__(self, name, location, angle, look_at, sky=(0, 1, 0)):
            self.name = name
            self.location = location
            self.angle = angle
            self.look_at = look_at
            self.sky = sky
            super().__init__()

        def update_script(self):
            location = self._position2str(self.location)
            look_at = self._position2str(self.look_at)
            sky = self._position2str(self.sky)
            cmds = []
            cmds.append("camera{")
            cmds.append(f"    location {location}")
            cmds.append(f"    angle {self.angle}")
            cmds.append(f"    look_at {look_at}")
            cmds.append(f"    sky {sky}")
            cmds.append("    right x*image_width/image_height")
            cmds.append("}")
            self.str = "\n".join(cmds)

    class Light(StageObject):
        """Light object

        Attributes
        ----------
        position : list or tuple
            Position vector of light location. (length=3)
        color : str or list
            Color of the light.
            Both string form of color or rgb (normalized) form is supported.
            Example) color='White', color=[1,1,1]
        """

        def __init__(self, position, color):
            self.position = position
            self.color = color
            super().__init__()

        def update_script(self):
            position = self._position2str(self.position)
            color = self._color2str(self.color)
            cmds = []
            cmds.append("light_source{")
            cmds.append(f"    {position}")
            cmds.append(f"    color {color}")
            cmds.append("}")
            self.str = "\n".join(cmds)

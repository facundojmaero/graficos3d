#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status
from itertools import cycle
import sys

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import pyassimp                     # 3D resource loader
import pyassimp.errors              # Assimp error management + exceptions

from transform import Trackball, identity, translate, scale, rotate, lerp, vec
from bisect import bisect_left      # search sorted keyframe lists

from PIL import Image               # load images for textures
from itertools import cycle

# -------------- OpenGL Texture Wrapper ---------------------------------------

class Texture:
    """ Helper class to create and automatically destroy textures """

    def __init__(self, file, wrap_mode=GL.GL_REPEAT, min_filter=GL.GL_LINEAR,
                 mag_filter=GL.GL_LINEAR_MIPMAP_LINEAR):
        self.glid = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.glid)
        # helper array stores texture format for every pixel size 1..4
        format = [GL.GL_LUMINANCE,
                  GL.GL_LUMINANCE_ALPHA, GL.GL_RGB, GL.GL_RGBA]
        try:
            # imports image as a numpy array in exactly right format
            tex = np.array(Image.open(file))
            format = format[0 if len(tex.shape) == 2 else tex.shape[2] - 1]
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, tex.shape[1],
                            tex.shape[0], 0, format, GL.GL_UNSIGNED_BYTE, tex)

            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap_mode)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap_mode)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, min_filter)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, mag_filter)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            message = 'Loaded texture %s\t(%s, %s, %s, %s)'
            print(message % (file, tex.shape, wrap_mode, min_filter, mag_filter))
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % file)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)

# -------------- Example texture plane class ----------------------------------
TEXTURE_VERT = """#version 330 core
uniform mat4 modelviewprojection;
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 vertexUV;

out vec2 fragTexCoord;
void main() {
    gl_Position = modelviewprojection * vec4(position, 1);
    //fragTexCoord = position.xy;
    fragTexCoord = vertexUV;
}"""

TEXTURE_FRAG = """#version 330 core
uniform sampler2D diffuseMap;
in vec2 fragTexCoord;
out vec4 outColor;
void main() {
    outColor = texture(diffuseMap, fragTexCoord);
}"""

# ------------- Node Hierarchical class ------------------------------------

class Node:
    """ Scene graph transform and parameter broadcast node """

    def __init__(self, name='', children=(), transform=identity(), **param):
        self.transform, self.param, self.name = transform, param, name
        self.children = list(iter(children))

    def add(self, *drawables):
        """ Add drawables to this node, simply updating children list """
        self.children.extend(drawables)

    def draw(self, projection, view, model, **param):
        """ Recursive draw, passing down named parameters & model matrix. """
        # merge named parameters given at initialization with those given here
        param = dict(param, **self.param)
        # model = ...   # what to insert here for hierarchical update?
        model = model @ self.transform
        for child in self.children:
            child.draw(projection, view, model, **param)

    def debug_children(self, tab):
        print(tab*' ' + self.name)
        for child in self.children:
            if child.__class__ == Node or child.__class__ == RotationControlNode:
                child.debug_children(tab+2)

class RotationControlNode(Node):
    def __init__(self, key_up, key_down, axis, angle=0, **param):
        super().__init__(**param)   # forward base constructor named arguments
        self.angle, self.axis = angle, axis
        self.key_up, self.key_down = key_up, key_down
        self.transform_backup = self.transform

    def draw(self, projection, view, model, win=None, **param):
        assert win is not None
        self.angle += 0.1 * int(glfw.get_key(win, self.key_up) == glfw.PRESS)
        self.angle -= 0.1 * int(glfw.get_key(win, self.key_down) == glfw.PRESS)
        self.transform = rotate(
            axis=self.axis, angle=self.angle) @ self.transform_backup

        # call Node's draw method to pursue the hierarchical tree calling
        super().draw(projection, view, model, win=win, **param)

    def debug_children(self, tab):
        print(tab*' ' + self.name)
        for child in self.children:
            if child.__class__ == Node or child.__class__ == RotationControlNode:
                child.debug_children(tab+2)

# -------------- Interpolator ----------------------------------------------

class KeyFrames:
    """ Stores keyframe pairs for any value type with interpolation_function"""

    def __init__(self, time_value_pairs, interpolation_function=lerp):
        if isinstance(time_value_pairs, dict):  # convert to list of pairs
            time_value_pairs = time_value_pairs.items()
        keyframes = sorted(((key[0], key[1]) for key in time_value_pairs))
        self.times, self.values = zip(*keyframes)  # pairs list -> 2 lists
        self.interpolate = interpolation_function

    def value(self, time):
        """ Computes interpolated value from keyframes, for a given time """

        # 1. ensure time is within bounds else return boundary keyframe
        if(time < self.times[0] or self.times[-1] < time):
            return self.values[-1]
        
        # 2. search for closest index entry in self.times, using bisect_left function
        closest_index = bisect_left(self.times, time)
        # 3. using the retrieved index, interpolate between the two neighboring values
        # in self.values, using the initially stored self.interpolate function
        return self.interpolate(self.values[closest_index-1], self.values[closest_index], 0.5)

# ------------ low level OpenGL object wrappers ----------------------------
class Shader:
    """ Helper class to create and automatically destroy shader program """
    @staticmethod
    def _compile_shader(src, shader_type):
        src = open(src, 'r').read() if os.path.exists(src) else src
        src = src.decode('ascii') if isinstance(src, bytes) else src
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        src = ('%3d: %s' % (i+1, l) for i, l in enumerate(src.splitlines()))
        if not status:
            log = GL.glGetShaderInfoLog(shader).decode('ascii')
            GL.glDeleteShader(shader)
            src = '\n'.join(src)
            print('Compile failed for %s\n%s\n%s' % (shader_type, log, src))
            return None
        return shader

    def __init__(self, vertex_source, fragment_source):
        """ Shader can be initialized with raw strings or source file names """
        self.glid = None
        vert = self._compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        if vert and frag:
            self.glid = GL.glCreateProgram()  # pylint: disable=E1111
            GL.glAttachShader(self.glid, vert)
            GL.glAttachShader(self.glid, frag)
            GL.glLinkProgram(self.glid)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            status = GL.glGetProgramiv(self.glid, GL.GL_LINK_STATUS)
            if not status:
                print(GL.glGetProgramInfoLog(self.glid).decode('ascii'))
                GL.glDeleteProgram(self.glid)
                self.glid = None

    def __del__(self):
        GL.glUseProgram(0)
        if self.glid:                      # if this is a valid shader object
            GL.glDeleteProgram(self.glid)  # object dies => destroy GL object

class VertexArray:
    """helper class to create and self destroy vertex array objects."""
    def __init__(self, attributes, index=None, usage=GL.GL_STATIC_DRAW):
        """ Vertex array from attributes and optional index array. Vertex
            Attributes should be list of arrays with one row per vertex. """

        # create vertex array object, bind it
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []  # we will store buffers in a list
        nb_primitives, size = 0, 0

        # load a buffer per initialized vertex attribute (=dictionary)
        for loc, data in enumerate(attributes):
            if data is None:
                continue

            # bind a new vbo, upload its data to GPU, declare its size and type
            self.buffers += [GL.glGenBuffers(1)]
            data = np.array(data, np.float32, copy=False)
            nb_primitives, size = data.shape
            GL.glEnableVertexAttribArray(loc)  # activates for current vao only
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data, usage)
            GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, 0, None)

        # optionally create and upload an index buffer for this object
        self.draw_command = GL.glDrawArrays
        self.arguments = (0, nb_primitives)
        if index is not None:
            self.buffers += [GL.glGenBuffers(1)]
            index_buffer = np.array(index, np.int32, copy=False)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer, usage)
            self.draw_command = GL.glDrawElements
            self.arguments = (index_buffer.size, GL.GL_UNSIGNED_INT, None)

        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def draw(self, primitive):
        """draw a vertex array, either as direct array or indexed array"""
        GL.glBindVertexArray(self.glid)
        self.draw_command(primitive, *self.arguments)
        GL.glBindVertexArray(0)

    def __del__(self):  # object dies => kill GL array and buffers from GPU
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(len(self.buffers), self.buffers)

# ------------  simple color fragment shader demonstrated in Practical 1 ------
COLOR_VERT = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 1) in vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 normal;
out vec3 fragColor;
out vec3 fragPos;

void main() {
    gl_Position = projection * view * model * vec4(position, 1);
    fragColor = color;
    normal = mat3(transpose(inverse(model))) * Normal;
    fragPos = vec3(model * vec4(position, 1.0f));
}"""


COLOR_FRAG = """#version 330 core
in vec3 fragColor;
in vec3 normal;
in vec3 fragPos;

out vec4 outColor;

uniform vec3 lightDir;
uniform float s;
uniform vec3 Ka;
uniform vec3 Kd;
uniform vec3 Ks;

void main() {
    //ambient
    vec3 ambient = Ka;

    //diffuse
    vec3 norm = normalize(normal);
    vec3 lightDir_norm = normalize(-lightDir);
    float result = max(dot(norm, lightDir_norm), 0.0);
    vec3 diffuse = result * Kd;
    diffuse = diffuse / pow( distance(fragPos, lightDir), 2);

    //specular
    vec3 viewDir = normalize(fragPos);
    vec3 reflectDir = normalize(reflect(-lightDir_norm, norm));
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), s);
    vec3 specular =  spec * Ks;
    specular = specular / pow( distance(fragPos, lightDir) ,2);
    
    outColor = vec4(diffuse + specular + ambient, 1);
}"""


# ------------  Scene object classes ------------------------------------------

class PhongMesh:

    def __init__(self, attributes, index=None):
        self.vertex_array = VertexArray(attributes, index)

    def draw(self, projection, view, model, color_shader=None, K_d=(1, 1, 1),
             K_a=(0, 0, 0), K_s=(1, 1, 1), s=16., **params):
             
        names = ['view', 'projection', 'model', 'lightDir', 's', 'Kd', 'Ka', 'Ks']
        loc = {n: GL.glGetUniformLocation(color_shader.glid, n) for n in names}
        GL.glUseProgram(color_shader.glid)

        GL.glUniformMatrix4fv(loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(loc['model'], 1, True, model)
        # direccion de fuente de luz
        GL.glUniform3fv(loc['lightDir'], 1, (0.0, 0.0, -2))
        GL.glUniform1f(loc['s'], s)        
        GL.glUniform3fv(loc['Ka'], 1, K_a)
        GL.glUniform3fv(loc['Kd'], 1, K_s)
        GL.glUniform3fv(loc['Ks'], 1, K_d)        

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        self.vertex_array.draw(GL.GL_TRIANGLES)


class TexturedMesh:
    """ Simple first textured object """

    def __init__(self, file, vertices, faces):

        self.file = file

        # feel free to move this up in the viewer as per other practicals
        self.shader = Shader(TEXTURE_VERT, TEXTURE_FRAG)

        self.vertex_array = VertexArray([vertices[0], vertices[1]], faces)

        # interactive toggles
        self.wrap = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                           GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filter = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                             (GL.GL_LINEAR, GL.GL_LINEAR),
                             (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap_mode, self.filter_mode = next(self.wrap), next(self.filter)

        # setup texture and upload it to GPU
        self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

        self.f6 = 0
        self.f7 = 0

    def draw(self, projection, view, model, win=None, **_kwargs):

        # some interactive elements
        if glfw.get_key(win, glfw.KEY_F6) == glfw.PRESS:
            self.f6 += 1
            if(self.f6 % 100 == 0):
                self.f6 = 0
                self.wrap_mode = next(self.wrap)
                self.texture = Texture(
                    self.file, self.wrap_mode, *self.filter_mode)

        if glfw.get_key(win, glfw.KEY_F7) == glfw.PRESS:
            self.f7 += 1
            if(self.f7 % 100 == 0):
                self.f7 = 0
                self.filter_mode = next(self.filter)
                self.texture = Texture(
                    self.file, self.wrap_mode, *self.filter_mode)

        GL.glUseProgram(self.shader.glid)

        # projection geometry
        loc = GL.glGetUniformLocation(self.shader.glid, 'modelviewprojection')
        GL.glUniformMatrix4fv(loc, 1, True, projection @ view @ model)

        # texture access setups
        loc = GL.glGetUniformLocation(self.shader.glid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc, 0)
        self.vertex_array.draw(GL.GL_TRIANGLES)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)

# ------------  Viewer class & window management ------------------------------
class GLFWTrackball(Trackball):
    """ Use in Viewer for interactive viewpoint control """

    def __init__(self, win):
        """ Init needs a GLFW window handler 'win' to register callbacks """
        super().__init__()
        self.mouse = (0, 0)
        glfw.set_cursor_pos_callback(win, self.on_mouse_move)
        glfw.set_scroll_callback(win, self.on_scroll)

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.zoom(deltay, glfw.get_window_size(win)[1])

class Viewer:
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=640, height=480):

        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.1, 0.1, 0.1, 0.1)
        GL.glEnable(GL.GL_DEPTH_TEST)         # depth test now enabled (TP2)
        GL.glEnable(GL.GL_CULL_FACE)          # backface culling enabled (TP2)

        # compile and initialize shader programs once globally
        self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)

        # initially empty list of object to draw
        self.drawables = []

        # initialize trackball
        self.trackball = GLFWTrackball(self.win)

        # cyclic iterator to easily toggle polygon rendering modes
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

        self.s = 1
        
    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer and depth buffer (<-TP2)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            winsize = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(winsize)
            
            # draw our scene objects
            for drawable in self.drawables:
                drawable.draw(projection, view, identity(),
                              color_shader=self.color_shader, win=self.win, s=self.s)

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables):
        """ add objects to draw in this window """
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))

class Cylinder(Node):
    """ Very simple cylinder based on practical 2 load function """

    def __init__(self):
        super().__init__()
        self.add(*load('cylinder.obj'))  # just load the cylinder from file

# ---------------- Shape constructors ----------------------------------------


def robot_arm():
    # construct our robot arm hierarchy for drawing in viewer
    cylinder = Cylinder()             # re-use same cylinder instance
    limb_shape = Node(transform=scale(0.1, 0.5, 0.1),
                      name='limb shape')  # make a thin cylinder
    limb_shape.add(cylinder)          # common shape of arm and forearm

    forearm_node = Node(transform=translate(0, 0.5, 0), name='forearm node')

    forearm_node.add(limb_shape)

    rot_forearm_node = RotationControlNode(glfw.KEY_LEFT, glfw.KEY_RIGHT, (
        1, 0, 0), transform=rotate((1, 0, 0), 45), children=[forearm_node], name='rot forearm node')

    move_forearm_node = Node(transform=translate(0, 1, 0), children=[
                             rot_forearm_node], name='move forearm node')

    # robot arm rotation with phi angle
    arm_node = Node(transform=translate(0, .5, 0), name='arm node')
    arm_node.add(limb_shape)

    rot_arm_node = RotationControlNode(glfw.KEY_UP, glfw.KEY_DOWN, (1, 0, 0), children=[arm_node, move_forearm_node], name='rot arm node')
    move_arm_node = Node(children=[rot_arm_node], name='move arm node')

    base_shape_size = Node(transform=scale(.5,.1,.5), children=[cylinder], name='base shape size')
    base_shape_rot = RotationControlNode(glfw.KEY_P, glfw.KEY_O, (0,1,0), transform=rotate((0,0,1),0), children=[base_shape_size, move_arm_node], name='base rotation')

    root_node = Node(children=[base_shape_rot], name='base shape')

    return root_node

# -------------- 3D textured mesh loader ---------------------------------------

def load_textured(file):
    """ load resources using pyassimp, return list of TexturedMeshes """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []  # error reading => return empty list

    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file)
    path = os.path.join('.', '') if path == '' else path
    for mat in scene.materials:
        mat.tokens = dict(reversed(list(mat.properties.items())))
        if 'file' in mat.tokens:  # texture file token
            tname = mat.tokens['file'].split('/')[-1].split('\\')[-1]
            # search texture in file's whole subdir since path often screwed up
            tname = [os.path.join(d[0], f) for d in os.walk(path) for f in d[2]
                     if tname.startswith(f) or f.startswith(tname)]
            if tname:
                mat.texture = tname[0]
            else:
                print('Failed to find texture:', tname)

    # prepare textured mesh
    meshes = []
    for mesh in scene.meshes:

        try:
            texture = scene.materials[mesh.materialindex].texture

            # tex coords in raster order: compute 1 - y to follow OpenGL convention
            tex_uv = ((0, 1) + mesh.texturecoords[0][:, :2] * (1, -1)
                    if mesh.texturecoords.size else None)

            # create the textured mesh object from texture, attributes, and indices
            meshes.append(TexturedMesh(texture, [mesh.vertices, tex_uv], mesh.faces))

        except:
            print("Error, no texture found")

            mat = scene.materials[mesh.materialindex].tokens
            node = Node(K_d=mat.get('diffuse', (1, 1, 1)),
                        K_a=mat.get('ambient', (0, 0, 0)),
                        K_s=mat.get('specular', (1, 1, 1)),
                        s=mat.get('shininess', 16.))
            node.add(PhongMesh([mesh.vertices, mesh.normals], mesh.faces))
            meshes.append(node)


    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes


# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    # place instances of our basic objects
    # viewer.add(*[mesh for file in sys.argv[1:] for mesh in load_textured(file)])

    for file in sys.argv[1:]:
        for mesh in load_textured(file):
            # viewer.add(mesh)
            node = Node(children=[mesh])
            rotation_node = RotationControlNode(glfw.KEY_P, glfw.KEY_O, (0, 1, 0), 
                                                children=[node], transform=rotate((0, 0, 1), 0))
            viewer.add(rotation_node)

    if len(sys.argv) < 2:
        print('Usage:\n\t%s [3dfile]*\n\n3dfile\t\t the filename of a model in'
              ' format supported by pyassimp.' % (sys.argv[0],))

    # esfera = Node(children=[*load_textured("bunny/bunny.obj")])
    # node = RotationControlNode(glfw.KEY_P, glfw.KEY_O, (0,1,0), children=[esfera], transform=rotate((0,0,1),0))
    # viewer.add(node)


    viewer.run()

    # my_keyframes = KeyFrames({0: 1, 3: 7, 6: 20})
    # print(my_keyframes.value(1.5))

    # vector_keyframes = KeyFrames({0: vec(1, 0, 0), 3: vec(0, 1, 0), 6: vec(0, 0, 1)})
    # print(vector_keyframes.value(1.5))   # should display numpy vector (0.5, 0.5, 0)

if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts

#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args

from transform import translate, rotate, scale, vec
from transform import frustum, perspective, Trackball, identity

import pyassimp                     # 3D ressource loader
import pyassimp.errors              # assimp error management + exceptions


# -------------- 3D ressource loader -----------------------------------------
def load(file):
    """ load resources from file using pyassimp, return list of ColorMesh """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []     # error reading => return empty list

    meshes = [ColorMesh([m.vertices, m.normals], m.faces) for m in scene.meshes]

    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    
    return meshes

#------------------------- ColorMesh Class -------------------------------------
"""The provided loading code is based on the assumption that you write a generic ColorMesh class 
(highlighted usage above) for which the initializer takes a list of attributes as first parameter, 
and an optional index buffer, just like the VertexArray class described above which it may invoke. 
The difference is that the draw() method of your new mesh class should invoke the color shader to draw 
the object whose attributes have been passed along, just like your Cylinder objects."""

class ColorMesh:
    def __init__(self, attributes, index=None):

        # attributes is a list of np.float32 arrays, index an optional np.uint32 array

        self.attributes = attributes

        self.glid = GL.glGenVertexArrays(1)  # create OpenGL vertex array id
        GL.glBindVertexArray(self.glid)      # activate to receive state below

        self.buffers = []
        self.index = None

        for layout_index, buffer_data in enumerate(attributes):
            # one vertex buffer per attribute
            self.buffers += [GL.glGenBuffers(1)]

            # bind the vbo, upload position data to GPU, declare its size and type
            GL.glEnableVertexAttribArray(layout_index)      # assign to layout = 0 attribute
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[layout_index])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, buffer_data, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(layout_index, len(buffer_data[0]), GL.GL_FLOAT, False, 0, None)

        if(index is not None):
            self.buffers += [GL.glGenBuffers(1)]                                           # create GPU index buffer
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])                  # make it active to receive
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index, GL.GL_STATIC_DRAW)          # our index array here
            self.index = index - 1

        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def draw(self, projection, view, model, color_shader, color):

        GL.glUseProgram(color_shader.glid)

        # my_color_location = GL.glGetUniformLocation(color_shader.glid, 'color')
        # GL.glUniform3fv(my_color_location, 1, self.attributes[1])                   # attributes[1] es el color

        matrix_location = GL.glGetUniformLocation(color_shader.glid, 'matrix')
        mat = projection @ view
        GL.glUniformMatrix4fv(matrix_location, 1, True, mat)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        GL.glBindVertexArray(self.glid)

        if(self.index is None):
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.attributes[0].shape[0])                 # attributes[0] es la posicion (lo uso sin indices)
        else:
            GL.glDrawElements(GL.GL_TRIANGLES, self.index.size, GL.GL_UNSIGNED_INT, None)        # lo uso si pase indices

        GL.glBindVertexArray(0)

    def __del__(self):
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(1, self.buffers)


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


# ------------  Simple color shaders ------------------------------------------
COLOR_VERT = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
uniform mat4 matrix;
out vec3 colorShader;
void main() {
    gl_Position = matrix * vec4(position, 1);
    colorShader = color;

}"""

COLOR_FRAG = """#version 330 core
out vec4 outColor;
uniform vec3 color;
in vec3 colorShader;
void main() {
    outColor = vec4(colorShader, 1);
    //outColor = vec4(vec3(.5,.5,.5), 1);

}"""

# ------------ Vertex Array Class ---------------------------------------------


class VertexArray:
    def __init__(self, attributes, index=None):
        # attributes is a list of np.float32 arrays, index an optional np.uint32 array

        self.attributes = attributes

        self.glid = GL.glGenVertexArrays(1)  # create OpenGL vertex array id
        GL.glBindVertexArray(self.glid)      # activate to receive state below

        self.buffers = []
        self.index = None

        for layout_index, buffer_data in enumerate(attributes):
            # one vertex buffer per attribute
            self.buffers += [GL.glGenBuffers(1)]

            # bind the vbo, upload position data to GPU, declare its size and type
            GL.glEnableVertexAttribArray(layout_index)      # assign to layout = 0 attribute
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[layout_index])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, buffer_data, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(layout_index, len(buffer_data[0]), GL.GL_FLOAT, False, 0, None)

        if(index is not None):
            self.buffers += [GL.glGenBuffers(1)]                                           # create GPU index buffer
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])                  # make it active to receive
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index, GL.GL_STATIC_DRAW)     # our index array here
            self.index = index


        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def draw(self, projection, view, model, color_shader, color):

        GL.glUseProgram(color_shader.glid)

        my_color_location = GL.glGetUniformLocation(color_shader.glid, 'color')
        GL.glUniform3fv(my_color_location, 1, self.attributes[1])                   # attributes[1] es el color

        matrix_location = GL.glGetUniformLocation(color_shader.glid, 'matrix')
        mat = projection @ view
        GL.glUniformMatrix4fv(matrix_location, 1, True, mat)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        GL.glBindVertexArray(self.glid)

        if(self.index is None):
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.attributes[0].shape[0])                 # attributes[0] es la posicion (lo uso sin indices)
        else:
            GL.glDrawElements(GL.GL_TRIANGLES, self.index.size, GL.GL_UNSIGNED_INT, None)        # lo uso si pase indices

        GL.glBindVertexArray(0)

    def __del__(self):
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(1, self.buffers)




# ------------  Viewer class & window management ------------------------------


class GLFWTrackball(Trackball):

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

        self.trackball = GLFWTrackball(self.win)

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

        # compile and initialize shader programs once globally
        self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)

        # initially empty list of object to draw
        self.drawables = []

        # Declare color for triangle
        self.color = [0.6, 0.6, 0.9]

        self.angle = 0

        GL.glEnable(GL.GL_DEPTH_TEST)

    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            # clear depth buffer
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # self.angle += 0.1

            winsize = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(winsize)

            # draw our scene objects
            for drawable in self.drawables:
                # drawable.draw(None, None, None, self.color_shader, self.color, self.angle)
                drawable.draw(projection, view, identity(), self.color_shader, self.color)

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
            elif key == glfw.KEY_A:
                self.color = np.add(self.color, [.1,0,0])
            elif key == glfw.KEY_S:
                self.color = np.add(self.color, [0,.1,0])
            elif key == glfw.KEY_D:
                self.color = np.add(self.color, [0,0,.1])
            elif key == glfw.KEY_Z:
                self.color = np.add(self.color, [-.1,0,0])
            elif key == glfw.KEY_X:
                self.color = np.add(self.color, [0,-.1,0])
            elif key == glfw.KEY_C:
                self.color = np.add(self.color, [0,0,-.1])



# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    # place instances of our basic objects

    # Piramide 1 (sin indices)

    vertices = np.array(((-0.5, 0, 0.5), (-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (0, 1, 0)), np.float32)

    position = np.array([
        vertices[0], vertices[1], vertices[2],
        vertices[0], vertices[2], vertices[3],
        vertices[0], vertices[4], vertices[3],
        vertices[0], vertices[4], vertices[1],
        vertices[1], vertices[4], vertices[2],
        vertices[3], vertices[4], vertices[2]])

    position += (-1, 0, 0)

    color = np.array((
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 1.0, 1.0)), np.float32)

    # viewer.add(VertexArray([position, color]))

    # Piramide 2 (con indices)

    position = np.array(
        ((-0.5, 0, 0.5), (-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (0, 1, 0)), np.float32)

    position += (1,0,0)

    index = np.array(((0,1,2), (0,2,3), (0,4,3), (0,4,1), (1,4,2), (3,4,2)), np.uint32)

    color = np.array((
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 1.0)), np.float32)

    # viewer.add(VertexArray([position, color], index))

    mesh = load("suzanne.obj")
    viewer.add(mesh[0])

    position = np.array((
        (1.000000, -1.000000, -1.000000),
        (1.000000, -1.000000, 1.000000),
        (-1.000000, -1.000000, 1.000000),
        (-1.000000, -1.000000, -1.000000),
        (1.000000, 1.000000, -0.999999),
        (0.999999, 1.000000, 1.000001),
        (-1.000000, 1.000000, 1.000000),
        (-1.000000, 1.000000, -1.000000))
        , np.float32)

    index = np.array(
        (1,1,1, 2,2,1, 3,3,1, 4,4,1,
        5,4,2, 8,1,2, 7,5,2, 6,3,2,
        1,2,3, 5,3,3, 6,4,3, 2,1,3,
        2,4,4, 6,1,4, 7,2,4, 3,3,4,
        3,1,5, 7,2,5, 8,3,5, 4,4,5,
        5,3,6, 1,4,6, 4,1,6, 8,2,6), np.uint32)
    
    index = index-1

    color = np.array((
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0)), np.float32)

    # viewer.add(VertexArray([position, color], index))

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts

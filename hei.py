import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import time

# Vertex shader source code
vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 aPos;
void main() {
    gl_Position = vec4(aPos, 1.0);
}
"""

# Fragment shader source code (Mandelbrot fractal with zoom and colors)
fragment_shader_source = """
#version 330 core
out vec4 FragColor;
uniform float time;

vec3 getColor(float t) {
    return vec3(0.5 + 0.5 * cos(6.28318 * (t + 0.0)), 
                0.5 + 0.5 * cos(6.28318 * (t + 0.33)), 
                0.5 + 0.5 * cos(6.28318 * (t + 0.67)));
}

void main() {
    vec2 c = (gl_FragCoord.xy / vec2(800.0, 600.0) - vec2(0.5, 0.5)) * 4.0 / pow(1.4, time) + vec2(-0.74364388703, 0.13182590421);
    vec2 z = vec2(0.0, 0.0);
    int max_iter = 100;
    int i;

    for (i = 0; i < max_iter; i++) {
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        if (dot(z, z) > 4.0) break;
    }

    float t = float(i) / float(max_iter);
    FragColor = vec4(getColor(t), 1.0);
}
"""

# Define the vertices of a quad
vertices = [
    -1.0, -1.0, 0.0,
     1.0, -1.0, 0.0,
     1.0,  1.0, 0.0,
    -1.0,  1.0, 0.0
]

# Define the indices for the quad (two triangles)
indices = [
    0, 1, 2,
    2, 3, 0
]

def main():
    # Initialize GLFW
    if not glfw.init():
        raise Exception("GLFW initialization failed")

    # Set OpenGL version to 3.3 Core
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(800, 600, "Mandelbrot Fractal with Zoom and Colors", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")

    # Make the window's context current
    glfw.make_context_current(window)

    # Set the viewport
    glViewport(0, 0, 800, 600)

    # Compile shaders and create shader program
    vertex_shader = compileShader(vertex_shader_source, GL_VERTEX_SHADER)
    fragment_shader = compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    shader_program = compileProgram(vertex_shader, fragment_shader)

    # Create a VAO and VBO
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)

    glBindVertexArray(vao)

    # Upload vertex data to the VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, np.array(vertices, dtype=np.float32), GL_STATIC_DRAW)

    # Upload index data to the EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, np.array(indices, dtype=np.uint32), GL_STATIC_DRAW)

    # Set up vertex attribute pointers
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), None)
    glEnableVertexAttribArray(0)

    # Unbind VAO
    glBindVertexArray(0)

    # Get the location of the time uniform
    time_uniform = glGetUniformLocation(shader_program, "time")

    # Start time
    start_time = time.time()

    # Main loop
    while not glfw.window_should_close(window):
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT)

        # Use the shader program and bind the VAO
        glUseProgram(shader_program)
        glBindVertexArray(vao)

        # Update the time uniform
        current_time = time.time() - start_time
        glUniform1f(time_uniform, current_time)

        # Draw the quad
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

    # Clean up
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    glDeleteBuffers(1, [ebo])
    glDeleteProgram(shader_program)

    # Terminate GLFW
    glfw.terminate()

if __name__ == "__main__":
    main()

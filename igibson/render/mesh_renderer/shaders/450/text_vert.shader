#version 450
layout (location = 0) in vec4 vertex; // contains pos then tex as vec2
out vec2 TexCoords;

// Indicates whether this is the foreground or background (F for text, B for quad)
uniform float background = 0.0;
uniform mat4 projection;

void main()
{
    gl_Position = projection * vec4(vertex.xy, -0.1 * background, 1.0);
    TexCoords = vertex.zw;
}  
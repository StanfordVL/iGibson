#version 410
layout (location = 0) in vec4 vertex; // contains pos then tex as vec2
out vec2 TexCoords;

uniform mat4 projection;

void main()
{
    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
    TexCoords = vertex.zw;
}  
#version 410 core
in vec3 position;
out vec3 texCoord;

uniform mat4 V;
uniform mat4 P;

void main()
{
	texCoord = position;
	gl_Position   = P * V  * vec4(position, 1.0);
}

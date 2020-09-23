#version 410 core

in vec3 texCoord;
uniform samplerCube envTexture;

layout (location = 0) out vec4 outputColour;
layout (location = 1) out vec4 NormalColour;
layout (location = 2) out vec4 InstanceColour;
layout (location = 3) out vec4 PCColour;

void main (void) {
    outputColour = texture(envTexture, texCoord);
    NormalColour = vec4(0,0,0,1);
    InstanceColour = vec4(0,0,0,1);
    PCColour = vec4(0,0,0,1);

}

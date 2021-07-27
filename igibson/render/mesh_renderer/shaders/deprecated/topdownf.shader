#version 450
uniform sampler2D texUnit;
uniform float use_texture;
in vec2 theCoords;
in vec3 Normal;
in vec3 Normal_cam;
in vec3 FragPos;
in vec3 Instance_color;
in vec3 Pos_cam;
in vec3 Diffuse_color;

layout (location = 0) out vec4 outputColour;
layout (location = 1) out vec4 NormalColour;
layout (location = 2) out vec4 InstanceColour;
layout (location = 3) out vec4 PCColour;

uniform vec3 light_position;  // in world coordinate
uniform vec3 light_color; // light color

void main() {
    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * light_color;
    vec3 lightDir = normalize(light_position - FragPos);
    float diff = 0.999 + 0.001 * max(dot(Normal, lightDir), 0.0);
    vec3 diffuse = diff * light_color;

    if (use_texture == 1) {
        outputColour =texture(texUnit, theCoords);// albedo only
    } else {
        outputColour = vec4(Diffuse_color,1) * diff; //diffuse color
    }

    NormalColour =  vec4((Normal_cam + 1) / 2,1);
    InstanceColour = vec4(Instance_color,1);
    PCColour = vec4(Pos_cam,1);
}
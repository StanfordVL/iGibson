#version 450
uniform sampler2D texUnit;
uniform sampler2D semUnit;
uniform sampler2D insUnit;

uniform float use_texture;
uniform float use_sem;
in vec2 theCoords;
in vec3 Normal_world;
in vec3 Normal_cam;
in vec3 FragPos;
in vec3 classId;
in vec3 instanceId;
in vec3 Pos_cam;
in vec3 Diffuse_color;

layout (location = 0) out vec4 outputColour;
layout (location = 1) out vec4 NormalColour;
layout (location = 2) out vec4 SemanticColour;
layout (location = 3) out vec4 PCColour;
layout (location = 4) out vec4 InstanceColour;

uniform vec3 light_position;  // in world coordinate
uniform vec3 light_color; // light color

void main() {
    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * light_color;
    vec3 lightDir = normalize(light_position - FragPos);
    float diff = 0.5 + 0.5 * max(dot(Normal_world, lightDir), 0.0);
    vec3 diffuse = diff * light_color;

    if (use_texture == 1) {
        outputColour = texture(texUnit, theCoords); // albedo only
    } else {
        outputColour = vec4(Diffuse_color, 1) * diff; //diffuse color
    }

    NormalColour =  vec4((Normal_cam + 1) / 2, 1);
    if (use_sem == 1) {
        SemanticColour = texture(semUnit, theCoords);
    } else {
        SemanticColour = vec4(classId, 1);
    }

    if (use_sem == 1) {
        InstanceColour = texture(insUnit, theCoords);
    } else {
        InstanceColour = vec4(instanceId, 1);
    }
    PCColour = vec4(Pos_cam, 1);
}
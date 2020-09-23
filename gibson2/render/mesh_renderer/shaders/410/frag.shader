#version 410
uniform sampler2D texUnit;
uniform sampler2D metallicTexture;
uniform sampler2D roughnessTexture;
uniform sampler2D normalTexture;

uniform samplerCube specularTexture;
uniform samplerCube irradianceTexture;
uniform sampler2D specularBRDF_LUT;
uniform vec3 eyePosition;

uniform float use_texture;
uniform float use_pbr;
uniform float use_pbr_mapping;
uniform float metallic;
uniform float roughness;

uniform sampler2D depthMap;
uniform int shadow_pass;

in vec2 theCoords;
in vec3 Normal_world;
in vec3 Normal_cam;
in vec3 FragPos;
in vec3 Instance_color;
in vec3 Pos_cam;
in vec3 Pos_cam_projected;
in vec3 Diffuse_color;
in mat3 TBN;
in vec4 FragPosLightSpace;

const float PI = 3.141592;
const float Epsilon = 0.00001;

layout (location = 0) out vec4 outputColour;
layout (location = 1) out vec4 NormalColour;
layout (location = 2) out vec4 InstanceColour;
layout (location = 3) out vec4 PCColour;

uniform vec3 light_position;  // in world coordinate
uniform vec3 light_color; // light color


void main() {
    vec3 lightDir = normalize(light_position - FragPos);
    float diff = 0.5 + 0.5 * max(dot(Normal_world, lightDir), 0.0);
    vec3 diffuse = diff * light_color;

    float shadow;
        if (shadow_pass == 2) {
            vec3 projCoords = FragPosLightSpace.xyz / FragPosLightSpace.w;
            projCoords = projCoords * 0.5 + 0.5;
            float closestDepth = texture(depthMap, projCoords.xy).b * 0.5 + 0.5;
            float currentDepth = projCoords.z;
            float bias = max(0.02 * (1.0 - dot(Normal_world, lightDir)), 0.002);
            shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;

            if ((projCoords.z > 1.0) || (projCoords.x > 1.0) || (projCoords.y > 1.0) || (projCoords.x < 0) || (projCoords.y <
             0))
                shadow = 0.0;
        }
        else {
            shadow = 0.0;
        }

    //not using pbr
    if (use_texture == 1) {
        outputColour = texture(texUnit, theCoords);// albedo only
    } else {
        outputColour = vec4(Diffuse_color,1) * diff; //diffuse color
    }

    NormalColour =  vec4((Normal_cam + 1) / 2,1);
    InstanceColour = vec4(Instance_color,1);
    if (shadow_pass == 1) {
        PCColour = vec4(Pos_cam_projected, 1);
    } else {
        PCColour = vec4(Pos_cam, 1);
    }
    outputColour = outputColour *  (1 - shadow * 0.5);

}

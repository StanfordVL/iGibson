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
uniform float use_two_light_probe;
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
in vec3 Pos_cam_prev;
in vec3 Pos_cam_projected;
in vec3 Diffuse_color;
in vec2 Optical_flow;
in mat3 TBN;
in vec4 FragPosLightSpace;

const float PI = 3.141592;
const float Epsilon = 0.00001;

layout (location = 0) out vec4 outputColour;
layout (location = 1) out vec4 NormalColour;
layout (location = 2) out vec4 InstanceColour;
layout (location = 3) out vec4 PCColour;
layout (location = 4) out vec4 SceneFlowColour;
layout (location = 5) out vec4 OpticalFlowColour;

uniform vec3 light_position;  // in world coordinate
uniform vec3 light_color; // light color


void main() {
    vec3 lightDir = vec3(0,0,1);//normalize(light_position);
    //sunlight pointing to z direction
    float diff = 0.5 + 0.5 * max(dot(Normal_world, lightDir), 0.0);
    vec3 diffuse = diff * light_color;
    vec2 texelSize = 1.0 / textureSize(depthMap, 0);

    float shadow;
        if (shadow_pass == 2) {
            vec3 projCoords = FragPosLightSpace.xyz / FragPosLightSpace.w;
            projCoords = projCoords * 0.5 + 0.5;
            float cosTheta = dot(Normal_world, lightDir);
            cosTheta = clamp(cosTheta, 0.0, 1.0);
            float Theta = acos(cosTheta);
            Theta = clamp(Theta, -PI/2.0+1e-4, PI/2.0-1e-4);
            float bias = 0.005*tan(Theta);
            bias = clamp(bias, 0.001 ,0.1);
            float currentDepth = projCoords.z;
            float closestDepth = 0;
            shadow = 0.0;
            float current_shadow = 0;
            for(int x = -1; x <= 1; ++x) // sample fewer to save time
            {
                for (int y = -1; y <= 1; ++y)
                {
                    closestDepth = texture(depthMap, projCoords.xy + vec2(x, y) * texelSize).b * 0.5 + 0.5;
                    current_shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
                    if ((projCoords.z > 1.0) || (projCoords.x > 1.0) || (projCoords.y > 1.0)
                    || (projCoords.x < 0) || (projCoords.y < 0)) current_shadow = 0.0;
                    shadow += current_shadow;
                }
            }
            shadow /= 9.0;

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
    SceneFlowColour =  vec4(Pos_cam - Pos_cam_prev,1);
    OpticalFlowColour =  vec4(Optical_flow,0,1);
}

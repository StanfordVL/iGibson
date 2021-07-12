#version 450
uniform sampler2D texUnit;
uniform sampler2D metallicTexture;
uniform sampler2D roughnessTexture;
uniform sampler2D normalTexture;

uniform samplerCube specularTexture;
uniform samplerCube irradianceTexture;
uniform sampler2D specularBRDF_LUT;

uniform samplerCube specularTexture2;
uniform samplerCube irradianceTexture2;
uniform sampler2D specularBRDF_LUT2;

uniform sampler2D lightModulationMap;

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
in mat3 TBN;
in vec4 FragPosLightSpace;
in vec2 Optical_flow;

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
float ndfGGX(float cosLh, float roughness)
{
	float alpha   = roughness * roughness;
	float alphaSq = alpha * alpha;

	float denom = (cosLh * cosLh) * (alphaSq - 1.0) + 1.0;
	return alphaSq / (PI * denom * denom);
}

// Single term for separable Schlick-GGX below.
float gaSchlickG1(float cosTheta, float k)
{
	return cosTheta / (cosTheta * (1.0 - k) + k);
}

// Schlick-GGX approximation of geometric attenuation function using Smith's method.
float gaSchlickGGX(float cosLi, float cosLo, float roughness)
{
	float r = roughness + 1.0;
	float k = (r * r) / 8.0; // Epic suggests using this roughness remapping for analytic lights.
	return gaSchlickG1(cosLi, k) * gaSchlickG1(cosLo, k);
}

// Shlick's approximation of the Fresnel factor.
vec3 fresnelSchlick(vec3 F0, float cosTheta)
{
	return F0 + (vec3(1.0) - F0) * pow(1.0 - cosTheta, 5.0);
}

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
        float bias = 0.005*tan(acos(cosTheta));
        bias = clamp(bias, 0.001 ,0.1);
        float currentDepth = projCoords.z;
        float closestDepth = 0;

        shadow = 0.0;
        float current_shadow = 0;

        for(int x = -2; x <= 2; ++x)
        {
            for (int y = -2; y <= 2; ++y)
            {
                closestDepth = texture(depthMap, projCoords.xy + vec2(x, y) * texelSize).b * 0.5 + 0.5;
                current_shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
                if ((projCoords.z > 1.0) || (projCoords.x > 1.0) || (projCoords.y > 1.0)
                || (projCoords.x < 0) || (projCoords.y < 0)) current_shadow = 0.0;
                shadow += current_shadow;
            }
        }
        shadow /= 25.0;
    }
    else {
        shadow = 0.0;
    }

    //not using pbr
    if (use_pbr == 0) {
        if (use_texture == 1) {
            outputColour = texture(texUnit, theCoords);// albedo only
        } else {
            outputColour = vec4(Diffuse_color,1) * diff; //diffuse color
        }
    }

    //use pbr, not using mapping
    if ((use_pbr == 1) && (use_pbr_mapping == 0)) {

        vec3 N = normalize(Normal_world);
        vec3 Lo = normalize(eyePosition - FragPos);
        float cosLo = max(0.0, dot(N, Lo));
        vec3 Lr = 2.0 * cosLo * N - Lo;

        vec3 albedo;
        if (use_texture == 1) {
            albedo = texture(texUnit, theCoords).rgb;// albedo only
        } else {
            albedo = Diffuse_color; //diffuse color
        }

        const vec3 Fdielectric = vec3(0.04);
        vec3 F0 = mix(Fdielectric, albedo, metallic);
		vec3 irradiance = texture(irradianceTexture, N).rgb;
		vec3 F = fresnelSchlick(F0, cosLo);
		vec3 kd = mix(vec3(1.0) - F, vec3(0.0), metallic);
		vec3 diffuseIBL = kd * albedo * irradiance;
		int specularTextureLevels = textureQueryLevels(specularTexture);
		vec3 specularIrradiance = textureLod(specularTexture, Lr, roughness * specularTextureLevels).rgb;
		vec2 specularBRDF = texture(specularBRDF_LUT, vec2(cosLo, roughness)).rg;
		vec3 specularIBL = (F0 * specularBRDF.x + specularBRDF.y) * specularIrradiance;
		vec3 ambientLighting = diffuseIBL + specularIBL;
        //vec3 reflection = textureLod(specularTexture, vec3(Lr.x, Lr.z, Lr.y), 1).rgb;
        outputColour = vec4(ambientLighting, 1);

    }

    // use pbr and mapping
    if ((use_pbr == 1) && (use_pbr_mapping == 1)) {
            vec3 normal_map = 2 * texture(normalTexture, theCoords).rgb - 1;
            mat3 TBN2 = TBN;
            if (!gl_FrontFacing) {
                TBN2 = mat3(TBN[0], TBN[1], -TBN[2]);
            }
            vec3 N = normalize(TBN2 * normal_map);

            vec3 Lo = normalize(eyePosition - FragPos);
            float cosLo = max(0.0, dot(N, Lo));
            vec3 Lr = 2.0 * cosLo * N - Lo;

            vec3 albedo;
            if (use_texture == 1) {
                albedo = texture(texUnit, theCoords).rgb;// albedo only
            } else {
                albedo = Diffuse_color; //diffuse color
            }

            float metallic_sampled = texture(metallicTexture, theCoords).r;
            float roughness_sampled = texture(roughnessTexture, theCoords).r;
            const vec3 Fdielectric = vec3(0.04);
            vec3 F0 = mix(Fdielectric, albedo, metallic_sampled);
    		vec3 irradiance;

            float modulate_factor;

            if (use_two_light_probe == 1) {
                vec4 room = texture(lightModulationMap, vec2((FragPos.x + 15.0)/30.0,
                (FragPos.y + 15.0)/30.0));
                modulate_factor = room.r;
            } else {
                modulate_factor = 1.0;
            }
            irradiance = texture(irradianceTexture, N).rgb * modulate_factor +
                (1-modulate_factor) * texture(irradianceTexture2, N).rgb;
    		vec3 F = fresnelSchlick(F0, cosLo);
    		vec3 kd = mix(vec3(1.0) - F, vec3(0.0), metallic_sampled);
    		vec3 diffuseIBL = kd * albedo * irradiance;
    		int specularTextureLevels = textureQueryLevels(specularTexture);
        	int specularTextureLevels2 = textureQueryLevels(specularTexture2);
    		vec3 specularIrradiance;
            vec2 specularBRDF;
            specularIrradiance = textureLod(specularTexture, Lr, roughness_sampled * specularTextureLevels).rgb *
            modulate_factor + textureLod(specularTexture2, Lr, roughness_sampled * specularTextureLevels2).rgb *
            (1-modulate_factor);

            specularBRDF = texture(specularBRDF_LUT, vec2(cosLo, roughness_sampled)).rg * modulate_factor +
            texture(specularBRDF_LUT2, vec2(cosLo, roughness_sampled)).rg * (1-modulate_factor);
    		vec3 specularIBL = (F0 * specularBRDF.x + specularBRDF.y) * specularIrradiance;
    		vec3 ambientLighting = diffuseIBL + specularIBL;
            //vec3 reflection = textureLod(specularTexture, vec3(Lr.x, Lr.z, Lr.y), 1).rgb;
            outputColour = vec4(ambientLighting, 1);
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

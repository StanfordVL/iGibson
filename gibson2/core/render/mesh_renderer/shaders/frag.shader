#version 450
uniform sampler2D texUnit;
uniform samplerCube specularTexture;
uniform samplerCube irradianceTexture;
uniform sampler2D specularBRDF_LUT;
uniform vec3 eyePosition;

uniform float use_texture;
uniform float use_pbr;
uniform float metalness;
uniform float roughness;


in vec2 theCoords;
in vec3 Normal_world;
in vec3 Normal_cam;
in vec3 FragPos;
in vec3 Instance_color;
in vec3 Pos_cam;
in vec3 Diffuse_color;

const float PI = 3.141592;
const float Epsilon = 0.00001;

layout (location = 0) out vec4 outputColour;
layout (location = 1) out vec4 NormalColour;
layout (location = 2) out vec4 InstanceColour;
layout (location = 3) out vec4 PCColour;

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
    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * light_color;
    vec3 lightDir = normalize(light_position - FragPos);
    float diff = 0.5 + 0.5 * max(dot(Normal_world, lightDir), 0.0);
    vec3 diffuse = diff * light_color;

    vec3 N = normalize(Normal_world);
    vec3 Lo = normalize(eyePosition - FragPos);
    float cosLo = max(0.0, dot(N, Lo));
    vec3 Lr = 2.0 * cosLo * N - Lo;

    if (use_texture == 1) {
        outputColour = texture(texUnit, theCoords);// albedo only
    } else {
        outputColour = vec4(Diffuse_color,1) * diff; //diffuse color
    }

    if (use_pbr == 1) {

        vec3 albedo = vec3(1,0.85,0);//texture(texUnit,theCoords).rgb;
        const vec3 Fdielectric = vec3(0.04);
        vec3 F0 = mix(Fdielectric, albedo, metalness);
		vec3 irradiance = texture(irradianceTexture, N).rgb;
		vec3 F = fresnelSchlick(F0, cosLo);
		vec3 kd = mix(vec3(1.0) - F, vec3(0.0), metalness);
		vec3 diffuseIBL = kd * albedo * irradiance;
		int specularTextureLevels = textureQueryLevels(specularTexture);
		vec3 specularIrradiance = textureLod(specularTexture, Lr, roughness * specularTextureLevels).rgb;
		vec2 specularBRDF = texture(specularBRDF_LUT, vec2(cosLo, roughness)).rg;
		vec3 specularIBL = (F0 * specularBRDF.x + specularBRDF.y) * specularIrradiance;
		vec3 ambientLighting = diffuseIBL + specularIBL;

        //vec3 reflection = textureLod(specularTexture, vec3(Lr.x, Lr.z, Lr.y), 1).rgb;
        outputColour = vec4(ambientLighting, 1);
    }
    NormalColour =  vec4((Normal_cam + 1) / 2,1);
    InstanceColour = vec4(Instance_color,1);
    PCColour = vec4(Pos_cam,1);
}
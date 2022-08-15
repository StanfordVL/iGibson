#version 460
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D s_color;
uniform sampler2D s_bloom;
uniform int postProcessingMode;  
uniform int eyeOffset; // 1: left, -1: right

float glareWidth = 0.7;
float c = 0.5; // constrast reduction factor
float t = 0.1; // tinting factor
vec3 TintColor = vec3(0.8, 0.5, 0); // tinting color
float exposure = 1.0; // bloom exposure

#define PI 3.1415926
// ======================== gaussian kernel ============================    
const float kernel[9] = float[](
    1.0 / 16, 2.0 / 16, 1.0 / 16,
    2.0 / 16, 4.0 / 16, 2.0 / 16,
    1.0 / 16, 2.0 / 16, 1.0 / 16  
);

const float offset = 1.0 / 300.0; 
const vec2 offsets[9] = vec2[](
    vec2(-offset,  offset), // top-left
    vec2( 0.0f,    offset), // top-center
    vec2( offset,  offset), // top-right
    vec2(-offset,  0.0f),   // center-left
    vec2( 0.0f,    0.0f),   // center-center
    vec2( offset,  0.0f),   // center-right
    vec2(-offset, -offset), // bottom-left
    vec2( 0.0f,   -offset), // bottom-center
    vec2( offset, -offset)  // bottom-right    
);

float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 lensflare(vec2 uv,vec2 pos)
{
	vec2 main = uv-pos;
	vec2 uvd = uv*(length(uv));
	
	float ang = atan(main.x,main.y);
	float dist=length(main); dist = pow(dist,.1);
	float n = rand(vec2(ang*16.0,dist*32.0));
	
	float f0 = 1.0/(length(uv-pos)*16.0+1.0);
	
	f0 = f0 + f0*(cos(sin(ang*.5+pos.x)*4.0 - cos(ang*.5+pos.y)*16.)*.1 + dist*glareWidth + .8);
	
	float f1 = max(0.01-pow(length(uv+1.2*pos),1.9),.0)*7.0;

	float f2 = max(1.0/(1.0+32.0*pow(length(uvd+0.8*pos),2.0)),.0)*00.25;
	float f22 = max(1.0/(1.0+32.0*pow(length(uvd+0.85*pos),2.0)),.0)*00.23;
	float f23 = max(1.0/(1.0+32.0*pow(length(uvd+0.9*pos),2.0)),.0)*00.21;
	
	vec2 uvx = mix(uv,uvd,-0.5);
	
	float f4 = max(0.01-pow(length(uvx+0.4*pos),2.4),.0)*6.0;
	float f42 = max(0.01-pow(length(uvx+0.45*pos),2.4),.0)*5.0;
	float f43 = max(0.01-pow(length(uvx+0.5*pos),2.4),.0)*3.0;
	
	uvx = mix(uv,uvd,-.4);
	
	float f5 = max(0.01-pow(length(uvx+0.2*pos),5.5),.0)*2.0;
	float f52 = max(0.01-pow(length(uvx+0.4*pos),5.5),.0)*2.0;
	float f53 = max(0.01-pow(length(uvx+0.6*pos),5.5),.0)*2.0;
	
	uvx = mix(uv,uvd,-0.5);
	
	float f6 = max(0.01-pow(length(uvx-0.3*pos),1.6),.0)*6.0;
	float f62 = max(0.01-pow(length(uvx-0.325*pos),1.6),.0)*3.0;
	float f63 = max(0.01-pow(length(uvx-0.35*pos),1.6),.0)*5.0;
	
	vec3 c = vec3(.0);
	
	c.r+=f2+f4+f5+f6; c.g+=f22+f42+f52+f62; c.b+=f23+f43+f53+f63;
	c = c*1.3 - vec3(length(uvd)*.05);
	c+=vec3(f0);
	
	return c;
}

vec3 cc(vec3 color, float factor, float factor2) // color modifier
{
	float w = color.x + color.y + color.z;
	return mix(color, vec3(w) * factor, w * factor2);
}


vec3 merge(vec3 oriColor, vec3 blendColor) {
    const float gamma = 2.2;
    oriColor += blendColor; // additive blending
    // tone mapping
    vec3 result = vec3(1.0) - exp(-oriColor * exposure);
    // also gamma correct while we're at it       
    result = pow(result, vec3(1.0 / gamma));
    return result;
}

void main()
{   
    vec3 texColor = texture(s_color, TexCoords).rgb;
    vec3 bloomColor = texture(s_bloom, TexCoords).rgb;

    vec2 center = vec2(0.5 - eyeOffset * 0.04, 0.5);
    // ======================== gaussian noise =============================
    float noise = sqrt(-2.0 * log(rand(TexCoords * 2))) * sin(2.0 * PI * rand(TexCoords)); // Box-Muller Transform

    switch (postProcessingMode) {
        case 0: // normal
            FragColor = vec4(texColor, 1.0);
            break;
        case 1: // Cataract
            // 1. blur
            vec3 sampleTex[9];
            for(int i = 0; i < 9; i++)
            {
                sampleTex[i] = vec3(texture(s_color, TexCoords.st + offsets[i]));
            }
            vec3 col = vec3(0.0);
            for(int i = 0; i < 9; i++)
                col += sampleTex[i] * kernel[i];
            // 2. reduce contrast
            vec3 tempColor = col * (1 - c) + vec3(0.5 * c);
            // 3. Color shift
            tempColor = tempColor * (1 - t) + TintColor * t;
            // 4. bloom
            // const float gamma = 2.2;
            // tempColor += bloomColor; // additive blending
            // // tone mapping
            // vec3 result = vec3(1.0) - exp(-tempColor * exposure);
            // // also gamma correct while we're at it       
            // result = pow(result, vec3(1.0 / gamma));
            FragColor = vec4(tempColor, 1.0);
            break;
        case 2: // AMD
        case 3: // Glaucoma
        case 4: // Presbyopia
        case 5: // myopia / hyperopia
            FragColor = vec4(texColor, 1.0);
            break;
        case 6: // ad hoc (light overexposure)
            vec3 color = vec3(1.4, 1.2, 1.0) * lensflare(TexCoords, center);
            color = cc(color, .5, .1);
            FragColor = vec4(mix(color, texColor, 0.5), 1.0);
            break;
    }
} 
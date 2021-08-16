#version 460
in vec2 TexCoords;
layout(location = 0) out vec4 color;

// Alpha value of the background - set to -1 if rendering text in foreground (default)
uniform float backgroundAlpha = -1.0;
uniform sampler2D text;
uniform vec3 textColor;
uniform vec3 backgroundColor;

void main()
{   
    // Sample and color text
    if (backgroundAlpha < 0) {
        vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);
        color = vec4(textColor, 1.0) * sampled;
    }
    // Sample and color background
    else {
        vec4 sampled = vec4(1.0, 1.0, 1.0, backgroundAlpha);
        color = vec4(backgroundColor, 1.0) * sampled;
    }
} 

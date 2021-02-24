import freetype as ft
from gibson2 import assets_path
import matplotlib.pyplot as plt
import numpy as np
import os


class Character(object):
    """
    Manages data for a single character.
    """
    def __init__(self, tex_id, size, bearing, advance, buffer):
        """
        :param tex_id: OpenGL texture id for this character
        :param size: list containing [x, y] size of character
        :param bearing: list containing [left, top] bearing of character
        :param advance: int representing offset to advance to get to next glyph
        :param buffer: raw character pixel values
        """
        self.tex_id = tex_id
        self.size = size
        self.bearing = bearing
        self.advance = advance
        self.buffer = buffer

    def viz_char(self):
        """
        Debug method for visualizing a character.
        """
        img_data = np.array(self.buffer)
        img_data = img_data.reshape((self.size[1], self.size[0]))
        plt.imshow(img_data, cmap='gray', vmin=0, vmax=255)
        plt.show()


class TextManager(object):
    """
    Manages character information and other centralized text info.
    """
    def __init__(self, renderer):
        """
        :param renderer: the renderer that will render the fonts
        """
        self.renderer = renderer
        if not self.renderer:
            raise ValueError('A renderer is required to render text')
        self.supported_fonts = ['OpenSans', 'SourceSansPro']
        self.supported_styles = ['Bold', 'Italic', 'Regular']
        self.font_folder = os.path.join(assets_path, 'fonts')
        # Font data is stored in dictionary - TODO: Keep this here? Change data type based on what is stored?
        self.font_data = {}

    def gen_text_fbo(self):
        """
        Generates the separate framebuffer used to render all text.
        """
        self.FBO, self.render_tex = self.renderer.r.genTextFramebuffer()

    def get_render_tex(self):
        """
        Returns the render texture belonging to the TextManager,
        which contains all Text that has the render_to_tex flag set to True.
        """
        return self.render_tex

    def load_font(self, font_name, font_style, font_size):
        """
        Loads font. TextManager stores data for each font, and only loads once for efficiency.
        Thus multiple Text objects can access the same font without loading it multiple times.
        :param font_name: name of font
        :param font_style: style of font (must be regular, italic or bold)
        :param font_size: vertical height of font letters in pixels
        """
        # Test if this font, style combination has already been loaded
        key = (font_name, font_style)
        if key in self.font_data:
            return self.font_data[key]

        if font_name not in self.supported_fonts:
            raise ValueError('Font {} not supported'.format(font_name))
        if font_style not in self.supported_styles:
            raise ValueError('Font style {} not supported'.format(font_style))
        font_path = os.path.join(self.font_folder, font_name, '{}-{}.ttf'.format(font_name, font_style))
        face = ft.Face(font_path)
        face.set_pixel_sizes(0, font_size)
        # Store fonts in dictionary mapping character code to Character object
        font_chars = {}
        # Load in all characters within the valid ASCII range (not SPACE or DEL)
        char_codes = [c[0] for c in face.get_chars() if c[0] > 32 and c[0] < 127]
        # Loop through characters and generate bitmaps
        for code in char_codes:
            face.load_char(code, flags=ft.FT_LOAD_RENDER)
            g = face.glyph
            bmap = g.bitmap
            # Convert list buffer into int numpy array so we can access raw data easily in C++ code
            buffer_data = np.ascontiguousarray(bmap.buffer, dtype=np.int32)
            tex_id = self.renderer.r.loadCharTexture(bmap.rows, bmap.width, buffer_data)
            next_c = Character(tex_id, [bmap.width, bmap.rows], [g.bitmap_left, g.bitmap_top], 
                               g.advance.x, bmap.buffer)
            font_chars[code] = next_c
        
        # Store font character dictionary in main font_data dictionary, under current font
        self.font_data[(font_name, font_style)] = font_chars
        return font_chars


class Text(object):
    """
    Text objects store information required to render a block of text.
    """
    def __init__(self,
                 text_data='PLACEHOLDER: PLEASE REPLACE!',
                 font_name='OpenSans',
                 font_style='Regular',
                 font_size=48,
                 color=[0, 0, 0],
                 pos=[0, 0],
                 scale=1.0,
                 render_to_tex=False,
                 background_color=None,
                 text_manager=None):
        """
        :param text_data: starting text to display (can be changed at a later time by set_text)
        :param font_name: name of font to render - same as font folder in iGibson assets
        :param font_size: size of font to render
        :param font_style: style of font - one of [regular, italic, bold]
        :param color: [r, g, b] color
        :param pos: [x, y] position of text box's bottom-left corner on screen, in pixels
        :param scale: scale factor for resizing text
        :param background_color: color of the background in form [r, g, b, a] - background will only appear if this is not None
        :param render_to_tex: whether text should be rendered to an OpenGL texture or the screen (the default)
        :param text_manager: TextManager object that handles raw character data for fonts
        """
        if not text_manager:
            raise ValueError('Each Text object requires a TextManager reference')
        self.font_name = font_name
        self.font_style = font_style
        # Note: font size is in pixels
        self.font_size = font_size
        self.render_to_tex = render_to_tex
        self.line_sep = 2 * self.font_size
        self.space_x = int(self.font_size / 2)
        # Text stores list of lines, which are each rendered
        self.set_text(text_data)
        self.set_attribs(pos=pos, scale=scale, color=color)
        self.background_color = background_color
        # Background margin in pixels (in both x and y directions)
        self.background_margin = 15
        # Text manager stores data for characters in a font
        self.tm = text_manager
        # Load font and extract character data
        self.char_data = self.tm.load_font(self.font_name, self.font_style, self.font_size)
        self.VAO, self.VBO = self.tm.renderer.r.setupTextRender()

    def set_text(self, input_text):
        """
        :param input_text: text to display - lines must be separated by the newline character
        """
        self.text = input_text.splitlines()

    def set_attribs(self, pos=None, scale=None, color=None):
        """
        Sets various text attributes.
        :param pos: [x, y] position of text box's bottom-left corner on screen, in pixels
        :param scale: scale factor for resizing text
        :param color: color of text
        """
        if pos:
            self.pos = pos
        if scale:
            self.scale = scale
        if color:
            self.color = color

    def render(self):
        """
        Render the current text object
        """
        # Text must be registered in order to render
        if not self.text:
            return
        if self.tm.renderer is None:
            return
        
        # Pass in -1 if we want to render to the screen
        self.tm.renderer.r.preRenderText(self.tm.renderer.textShaderProgram, self.tm.FBO if self.render_to_tex else -1, self.VAO, self.color[0], self.color[1], self.color[2])

        # Precalculate render data for each character so we can figure out the bounds of the background quad
        text_to_render = self.text[::-1]
        # Store characte render data as list of tuples (xpos, ypos, w, h, tex_id)
        char_render_data = []
        for i in range(len(text_to_render)):
            line = text_to_render[i]
            # x pos resets back to left each time a new line is rendered
            next_x = self.pos[0]
            next_y = self.pos[1] + i * self.line_sep
            for c in line:
                # Convert character to ASCII
                c = ord(c)
                # Deal with spaces - and add space for unrecognized characters
                if c <= 32 or c >= 127:
                    next_x += self.space_x
                    continue

                c_data = self.char_data[c]
                xpos = next_x + c_data.bearing[0] * self.scale
                ypos = next_y - (c_data.size[1] - c_data.bearing[1]) * self.scale
                w = c_data.size[0] * self.scale
                h = c_data.size[1] * self.scale

                char_render_data.append((xpos, ypos, w, h, c_data.tex_id))

                # Advance x position to next glyph - advance is stored in units of 1/64 pixels, so we need to divide by 64
                next_x += ((c_data.advance / 64.0) * self.scale)

        # Optionally render background first so alpha blending works correctly
        if self.background_color:
            # Find bottom-left corner
            min_x = sorted(char_render_data, key=lambda x: x[0])[0][0]
            min_y = sorted(char_render_data, key=lambda x: x[1])[0][1]
            bottom_left_x = min_x - self.background_margin
            bottom_left_y = min_y - self.background_margin
            # Find top-right corner
            max_x, max_y = 0.0, 0.0
            for d in char_render_data:
                letter_right = d[0] + d[2]
                letter_top = d[1] + d[3]
                if letter_right > max_x:
                    max_x = letter_right
                if letter_top > max_y:
                    max_y = letter_top
            top_right_x = max_x + self.background_margin
            top_right_y = max_y + self.background_margin
            b_w = top_right_x - bottom_left_x
            b_h = top_right_y - bottom_left_y

            # Unpack color data
            b_r, b_g, b_b, b_a = self.background_color
            # Render background
            self.tm.renderer.r.renderBackgroundQuad(bottom_left_x, bottom_left_y, b_w, b_h, self.VBO, 
                                                    self.tm.renderer.textShaderProgram, b_a, b_r, b_g, b_b)

        # Finally render all characters
        for r_data in char_render_data:
            xpos, ypos, w, h, tex_id = r_data
            self.tm.renderer.r.renderChar(xpos, ypos, w, h, tex_id, self.VBO)

        # Perform render clean-up
        self.tm.renderer.r.postRenderText()
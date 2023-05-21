import cv2
import numpy as np
# from storage import Object


class Drawer:
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
              (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
              (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
              (64, 64, 0), (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0),
              (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192), (128, 64, 0),
              (64, 128, 0), (0, 128, 64), (128, 0, 64), (64, 0, 128), (0, 64, 128),
              (192, 64, 0), (64, 192, 0), (0, 192, 64), (192, 0, 64), (64, 0, 192),
              (0, 64, 192), (128, 192, 0), (192, 128, 0), (0, 192, 128), (128, 0, 192),
              (192, 0, 128), (128, 64, 64), (64, 128, 64), (64, 64, 128), (128, 128, 64),
              (128, 64, 128), (64, 128, 128), (255, 64, 0), (64, 255, 0), (0, 255, 64),
              (255, 0, 64), (64, 0, 255), (0, 64, 255), (255, 128, 0), (128, 255, 0),
              (0, 255, 128), (255, 0, 128), (128, 0, 255), (0, 128, 255), (255, 192, 0),
              (192, 255, 0), (0, 255, 192), (255, 0, 192), (192, 0, 255), (0, 192, 255),
              (255, 255, 64), (255, 64, 255), (64, 255, 255), (255, 128, 128), (128, 255, 128),
              (128, 128, 255), (255, 255, 128), (255, 128, 255), (128, 255, 255), (255, 192, 192),
              (192, 255, 192), (192, 192, 255), (64, 255, 64), (255, 64, 64), (192, 255, 64), (64, 192, 255),
              (64, 64, 64), (128, 128, 128), (192, 192, 192), (92, 92, 205), (11, 134, 184), (140, 230, 240),
              (50, 205, 154), (47, 255, 143), (143, 188, 143), (79, 79, 47), (208, 224, 64), (180, 130, 70),
              (219, 112, 147), (204, 50, 153), (216, 191, 216), (30, 105, 210), (222, 196, 176),
              (255, 255, 255), (0, 0, 0)]

    def __init__(self, image=None, class2colors={}):
        self.image = image
        self.class2colors = class2colors
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        self.text_color = (255, 255, 255)
        self.text_background_color = (0, 0, 0)

    def draw_bbox(self, bbox, class_name, confidence=None, alpha=0.2, color=None):
        bbox = np.round(np.array(bbox)).astype(int)
        if not color:
            color = self.get_color(class_name)

        overlay = self.image.copy()
        cv2.rectangle(overlay, bbox[0]-1, bbox[1]+1, color, -1)
        self.image = cv2.addWeighted(overlay, alpha, self.image, 1 - alpha, 0)
        cv2.rectangle(self.image, bbox[0]-1, bbox[1]+1, color, 1)
        text = f"{class_name}"
        if confidence:
            text += f" {confidence:.2f}"

        # if polygon is not None:
        #     text += f" polygon ({polygon_confidence:.2f})"
        text_size, _ = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)
        text_x = bbox[0][0] + 5
        text_y = bbox[0][1] + text_size[1] + 5
        # cv2.rectangle(self.image, (bbox[0], bbox[1]), (text_x + text_size[0], text_y + text_size[1]),
        #               self.text_background_color, -1)
        cv2.putText(self.image, text, (text_x, text_y), self.font, self.font_scale, self.get_color(class_name+"_text"),
                    self.font_thickness, cv2.LINE_AA)
        return self

    def draw_polygon_area(self, polygon, class_name='', confidence=None, alpha=0.2, color=None):
        if not color:
            color = self.get_color(class_name)
        # Convert polygon to numpy array and round to integers
        polygon = np.round(np.array(polygon)).astype(int)

        # Draw polygon with transparency based on confidence
        overlay = self.image.copy()
        cv2.fillPoly(overlay, [polygon], color)
        # alpha = 0.5 * polygon_confidence
        self.image = cv2.addWeighted(overlay, alpha, self.image, 1 - alpha, 0)
        # Draw class name at polygon centroid
        text = f"{class_name}"
        if confidence:
            text += f" {confidence:.2f}"

        if text:
            moments = cv2.moments(polygon)
            cx = int(moments['m10'] / (moments['m00'] + 1e-10))
            cy = int(moments['m01'] / (moments['m00'] + 1e-10))
            text_size = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)[0]
            wh = polygon.max(axis=0) - polygon.min(axis=0)
            if text_size[1] < wh[1] + 5 and text_size[0] < wh[0] + 5:
                text_x = cx - text_size[0] // 2
                text_y = cy + text_size[1] // 2
                cv2.putText(self.image, text, (text_x, text_y), self.font, self.font_scale,
                            self.get_color(class_name+"_text"),
                            self.font_thickness, cv2.LINE_AA)
        return self

    def draw_polygon_border(self, polygon, class_name=None, color=None, alpha_border=0.8):
        if not color:
            color = self.get_color(class_name)
        polygon = np.round(np.array(polygon)).astype(int).reshape(-1, 1, 2)

        imgdraw = cv2.polylines(np.zeros_like(self.image, np.uint8), [polygon], isClosed=True, thickness=1, color=color)
        mask = imgdraw.astype(bool)
        self.image[mask] = cv2.addWeighted(imgdraw, alpha_border, self.image, 1 - alpha_border, 0)[mask]
        return self

    def draw_text(self, text='', font_scale=0.8, thickness=1, alpha=0.5):
        # Set up text parameters
        font_face = cv2.FONT_HERSHEY_COMPLEX
        padding = 5  # Padding around the text
        bonus_padding_background_down = 3

        # Get the text size
        text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)

        # Set the text position (top-right corner with padding)
        text_pos = (self.image.shape[1] - text_size[0] - padding, padding + text_size[1])

        # Draw the text background
        overlay = self.image.copy()

        cv2.rectangle(overlay, (text_pos[0] - padding, text_pos[1] - padding - text_size[1]),
                      (text_pos[0] + text_size[0] + padding, text_pos[1] + padding + bonus_padding_background_down),
                      self.text_background_color, -1)
        self.image = cv2.addWeighted(overlay, alpha, self.image, 1 - alpha, 0)
        # Draw the text itself
        cv2.putText(self.image, text, text_pos, font_face, font_scale, self.text_color, thickness, cv2.LINE_AA)
        return self

    def save_image(self, filename):
        cv2.imwrite(filename, self.image)

    def show(self, window=''):
        cv2.imshow(window, self.image)
        key = cv2.waitKey(0)
        if (key == 27):
            cv2.destroyAllWindows()

    def get_color(self, class_name):
        color = self.class2colors.get(class_name, None)
        if not color:
            if class_name.endswith("_text"):
                color = self.text_color
                self.class2colors[class_name] = self.text_color
            else:
                color = self.COLORS[len(self.class2colors)]
                self.class2colors[class_name] = color
        return color

    def set_image(self, image):
        self.image = image

#    def draw_object(self, obj: Object):
#         # custom draw object
#         if obj.class_name == "number":
#             rwy_num = str(obj.attributes.get('number', '')) + obj.attributes.get('symbol', '')
#             self.draw_bbox(obj.bbox, rwy_num, obj.confidence, alpha=0.1, color=self.get_color(obj.class_name))
#             return
#
#         if obj.class_name == 'thr':
#             if obj.polygon is not None:
#                 color = self.get_color(obj.class_name + '_poly')
#                 self.draw_polygon_area(obj.polygon, class_name=obj.class_name, confidence=obj.confidence,
#                                        alpha=0.3, color=color) \
#                     .draw_polygon_border(obj.polygon, alpha_border=0.9, color=color)
#             else:
#                 self.draw_bbox(obj.bbox, obj.class_name, obj.confidence, alpha=0.1)
#             return

        # default drawing

        # polygon
        # if obj.polygon is not None:
        #     color = self.get_color(obj.class_name + '_poly')
        #     self.draw_polygon_area(obj.polygon, alpha=0.2, color=color)\
        #         .draw_polygon_border(obj.polygon, alpha_border=0.9, color=color)
        # # bbox
        # self.draw_bbox(obj.bbox, obj.class_name, obj.confidence, alpha=0.1)


if __name__ == "__main__":
    img = cv2.imread(r"D:\temp\testimg.jpg")
    draw = Drawer(img)
    draw.draw_bbox((100, 100, 600, 600), "car", 0.86734, alpha=0.15)
    draw.draw_polygon_area([[300, 300], [300, 330], [330, 330], [330, 300]], 'dog', alpha=0.15)
    draw.draw_polygon_area([(350, 300), (450, 375), (450, 475), (350, 550), (250, 475), (250, 375)], 'hexagon', alpha=0.15)

    draw.draw_polygon_area([(350, 300), (450, 375), (450, 475), (350, 550), (250, 475), (250, 600)], 'hexagon2', alpha=0.15)

    draw.draw_polygon_area([(350, 300), (450, 375), (450, 475), (350, 550), (800, 475), (250, 600)], 'hexagon3', alpha=0.15)
    draw.draw_text("fps: 2132 | count objects: 3")
    draw.save_image('test.jpg')
    print(draw.class2colors)
    # draw.show()

    # def draw_palette(colors, width=50, height=50):
    #     num_colors = len(colors)
    #     palette = np.zeros(((num_colors // 10 + 1) * height, 10 * width, 3), dtype=np.uint8)
    #
    #     for i, color in enumerate(colors):
    #         row = i // 10
    #         col = i % 10
    #         start_x = col * width
    #         end_x = start_x + width
    #         start_y = row * height
    #         end_y = start_y + height
    #         palette[start_y:end_y, start_x:end_x, :] = color
    #
    #     return palette
    #
    #
    # palette = draw_palette(draw.COLORS)
    # cv2.imshow('Palette', palette)
    # cv2.waitKey(0)
    # draw.show()

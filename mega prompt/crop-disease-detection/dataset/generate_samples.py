"""
Generate realistic procedural sample leaf images for offline demonstration and testing.
Includes healthy leaves, diseased leaves with accurate visual patterns,
and a non-leaf object to demonstrate quality filter rejection.
"""
from PIL import Image, ImageDraw, ImageFilter
import math
import random
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "sample_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_base_canvas(bg_color=(235, 238, 233)):
    img = Image.new("RGB", (400, 400), bg_color)
    return img

def draw_leaf_shape(draw, center=(200, 200), length=160, width=90, angle=0, base_color=(58, 125, 48)):
    points = []
    steps = 40
    for i in range(steps + 1):
        t = i / steps
        # leaf parametric width
        w = math.sin(t * math.pi) * (width * 0.5)
        # x is along length, y is along width
        x = (t - 0.5) * length
        y = w
        # rotate
        rad = math.radians(angle)
        rx = center[0] + x * math.cos(rad) - y * math.sin(rad)
        ry = center[1] + x * math.sin(rad) + y * math.cos(rad)
        points.append((rx, ry))
        
    for i in range(steps, -1, -1):
        t = i / steps
        w = -math.sin(t * math.pi) * (width * 0.5)
        x = (t - 0.5) * length
        y = w
        rad = math.radians(angle)
        rx = center[0] + x * math.cos(rad) - y * math.sin(rad)
        ry = center[1] + x * math.sin(rad) + y * math.cos(rad)
        points.append((rx, ry))
        
    draw.polygon(points, fill=base_color, outline=(40, 90, 32))
    
    # Draw central vein
    rad = math.radians(angle)
    x1 = center[0] - (length * 0.5) * math.cos(rad)
    y1 = center[1] - (length * 0.5) * math.sin(rad)
    x2 = center[0] + (length * 0.5) * math.cos(rad)
    y2 = center[1] + (length * 0.5) * math.sin(rad)
    draw.line([(x1, y1), (x2, y2)], fill=(95, 160, 75), width=3)
    
    # Secondary veins
    for s in [0.25, 0.45, 0.65, 0.8]:
        vx = center[0] + (s - 0.5) * length * math.cos(rad)
        vy = center[1] + (s - 0.5) * length * math.sin(rad)
        for side in [-1, 1]:
            vrad = rad + side * math.radians(45)
            vx2 = vx + 35 * math.cos(vrad)
            vy2 = vy + 35 * math.sin(vrad)
            draw.line([(vx, vy), (vx2, vy2)], fill=(85, 145, 68), width=1)

def generate_healthy_tomato():
    img = create_base_canvas()
    draw = ImageDraw.Draw(img)
    # Compound tomato foliage
    draw_leaf_shape(draw, center=(200, 200), length=220, width=120, angle=-10, base_color=(46, 125, 50))
    draw_leaf_shape(draw, center=(130, 150), length=130, width=70, angle=-45, base_color=(56, 142, 60))
    draw_leaf_shape(draw, center=(270, 160), length=120, width=65, angle=35, base_color=(56, 142, 60))
    draw_leaf_shape(draw, center=(140, 260), length=125, width=68, angle=-70, base_color=(56, 142, 60))
    draw_leaf_shape(draw, center=(260, 270), length=120, width=65, angle=65, base_color=(56, 142, 60))
    # Soft texture
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img.save(OUTPUT_DIR / "tomato_healthy.jpg", "JPEG", quality=95)
    print("Generated tomato_healthy.jpg")

def generate_tomato_early_blight():
    img = create_base_canvas()
    draw = ImageDraw.Draw(img)
    draw_leaf_shape(draw, center=(200, 200), length=230, width=130, angle=-5, base_color=(58, 120, 52))
    
    # Concentric target lesions with yellow halos
    lesions = [(170, 180, 28), (220, 220, 36), (150, 230, 22), (230, 150, 26)]
    for x, y, r in lesions:
        # Yellow halo
        draw.ellipse([x - r - 8, y - r - 8, x + r + 8, y + r + 8], fill=(195, 175, 42))
        # Concentric dark rings
        for ring in range(r, 4, -5):
            c = (70, 42, 28) if (ring // 5) % 2 == 0 else (100, 65, 40)
            draw.ellipse([x - ring, y - ring, x + ring, y + ring], fill=c)
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(45, 25, 15))
        
    img = img.filter(ImageFilter.SMOOTH)
    img.save(OUTPUT_DIR / "tomato_early_blight.jpg", "JPEG", quality=95)
    print("Generated tomato_early_blight.jpg")

def generate_potato_late_blight():
    img = create_base_canvas()
    draw = ImageDraw.Draw(img)
    draw_leaf_shape(draw, center=(200, 200), length=220, width=135, angle=15, base_color=(52, 115, 48))
    
    # Large water-soaked purplish dark irregular blotches
    for bx, by in [(160, 160), (230, 230), (180, 240)]:
        draw.ellipse([bx - 35, by - 30, bx + 45, by + 40], fill=(170, 165, 60)) # chlorotic margin
        draw.ellipse([bx - 28, by - 24, bx + 35, by + 32], fill=(62, 50, 55))  # dark purplish rot
        draw.ellipse([bx - 18, by - 14, bx + 22, by + 20], fill=(40, 32, 35))  # necrotic core
        
    img = img.filter(ImageFilter.SMOOTH)
    img.save(OUTPUT_DIR / "potato_late_blight.jpg", "JPEG", quality=95)
    print("Generated potato_late_blight.jpg")

def generate_corn_rust():
    img = create_base_canvas()
    draw = ImageDraw.Draw(img)
    # Long strap-like maize leaf blade
    draw_leaf_shape(draw, center=(200, 200), length=320, width=85, angle=30, base_color=(68, 138, 55))
    
    # Powdery cinnamon-rust colored pustules scattered on leaf blade
    random.seed(42)
    for _ in range(70):
        px = random.randint(120, 280)
        py = int(200 + (px - 200) * math.tan(math.radians(30)) + random.randint(-22, 22))
        rx = random.randint(3, 7)
        ry = random.randint(2, 4)
        draw.ellipse([px - rx, py - ry, px + rx, py + ry], fill=(185, 78, 25), outline=(130, 50, 15))
        
    img = img.filter(ImageFilter.SMOOTH)
    img.save(OUTPUT_DIR / "corn_common_rust.jpg", "JPEG", quality=95)
    print("Generated corn_common_rust.jpg")

def generate_apple_scab():
    img = create_base_canvas()
    draw = ImageDraw.Draw(img)
    draw_leaf_shape(draw, center=(200, 200), length=210, width=125, angle=-20, base_color=(54, 120, 52))
    
    # Olive-green to velvety brown circular scabby spots
    for sx, sy in [(170, 160), (220, 180), (180, 220), (230, 240), (140, 210)]:
        draw.ellipse([sx - 16, sy - 16, sx + 16, sy + 16], fill=(95, 88, 45))
        draw.ellipse([sx - 11, sy - 11, sx + 11, sy + 11], fill=(68, 62, 32))
        draw.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=(42, 38, 20))
        
    img = img.filter(ImageFilter.SMOOTH)
    img.save(OUTPUT_DIR / "apple_scab.jpg", "JPEG", quality=95)
    print("Generated apple_scab.jpg")

def generate_non_leaf_sample():
    # A metallic blue tech circuit / non-plant pattern to demonstrate rejection filter
    img = Image.new("RGB", (400, 400), (45, 52, 70))
    draw = ImageDraw.Draw(img)
    # Metallic blue and grey geometric lines (circuit/keyboard)
    for i in range(10, 400, 40):
        draw.line([(0, i), (400, i)], fill=(30, 85, 140), width=2)
        draw.line([(i, 0), (i, 400)], fill=(30, 85, 140), width=2)
    draw.rectangle([100, 100, 300, 300], fill=(25, 35, 55), outline=(0, 160, 230), width=3)
    draw.text((130, 190), "METALLIC TEST OBJECT", fill=(0, 200, 255))
    img.save(OUTPUT_DIR / "non_leaf_sample.jpg", "JPEG", quality=95)
    print("Generated non_leaf_sample.jpg")

if __name__ == "__main__":
    generate_healthy_tomato()
    generate_tomato_early_blight()
    generate_potato_late_blight()
    generate_corn_rust()
    generate_apple_scab()
    generate_non_leaf_sample()
    print("All sample demonstration images successfully created in", OUTPUT_DIR)

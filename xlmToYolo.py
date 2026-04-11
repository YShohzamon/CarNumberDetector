import xml.etree.ElementTree as ET
import os


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


# XML joylashgan papka
xml_path = 'dataset/annotations'
# TXT saqlanadigan papka
out_path = 'dataset/labels'
os.makedirs(out_path, exist_ok=True)

for xml_file in os.listdir(xml_path):
    if not xml_file.endswith('.xml'): continue

    tree = ET.parse(os.path.join(xml_path, xml_file))
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(os.path.join(out_path, xml_file.replace('.xml', '.txt')), 'w') as f:
        for obj in root.iter('object'):
            # Klass nomi (masalan, 'licence' bo'lsa 0 deb belgilaymiz)
            cls_id = 0
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            f.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")

print("Konvertatsiya yakunlandi!")
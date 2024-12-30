# Create the YAML configuration file for training
def create_yaml():
    yaml_content = """
    # Ultralytics YOLO 🚀, AGPL-3.0 license
    # PASCAL VOC dataset http://host.robots.ox.ac.uk/pascal/VOC by University of Oxford
    # Example usage: yolo train data=VOC.yaml
    # parent
    # ├── ultralytics
    # └── datasets
    #     └── VOC  ← downloads here (2.8 GB)


    # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
    path: /content/datasets/VOC_l
    train: # train images (relative to 'path')  16551 images
      - images/train2012
    val: # val images (relative to 'path')  4952 images
      - images/val2012
    test: # test images (optional)
      - images/val2012

    # Classes
    names:
      0: airplane
      1: ship
      2: storage tank
      3: baseball diamond
      4: tennis court
      5: basketball court
      6: ground track field
      7: harbor
      8: bridge
      9: vehicle
    """

    with open("VOC_2012.yaml", 'w') as yaml_file:
        yaml_file.write(yaml_content)
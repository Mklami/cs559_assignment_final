import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (80, 80)

tf.random.set_seed(1337)

def parse_example(path):
    fname = tf.strings.split(path, '/')[-1]
    label_str = tf.strings.split(fname, '_')[0]
    label = tf.strings.to_number(label_str, out_type=tf.float32)
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.shape(img)
    needs_resize = tf.logical_or(
        tf.not_equal(shape[0], IMG_SIZE[0]),
        tf.not_equal(shape[1], IMG_SIZE[1])
    )
    img = tf.cond(
        needs_resize,
        lambda: tf.image.resize(img, IMG_SIZE),
        lambda: img
    )
    
    return img, label

def augment_image(img, label):
    """Apply GENTLE random augmentations to training images"""
    # Random horizontal flip
    img = tf.image.random_flip_left_right(img)
    
    # Random brightness (±10%)
    img = tf.image.random_brightness(img, max_delta=0.1)
    
    # Random contrast (±10%)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    
    # Random saturation (±20%)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    
    # REMOVE harsh 90-degree rotations
    # Instead, add small random rotations if needed (requires tfa or custom impl)
    
    # Ensure values stay in [0, 1]
    img = tf.clip_by_value(img, 0.0, 1.0)
    
    return img, label

def build_ds(pattern, batch=64, is_train=False, shuffle_buffer=10000, seed=42, augment=False):
    files = tf.data.Dataset.list_files(pattern, shuffle=is_train, seed=seed)
    ds = files.map(parse_example, num_parallel_calls=AUTOTUNE)
    
    ds = ds.cache()  

    if augment:
        ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)
    
    if is_train:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    
    
    return ds

# Quick sanity check
if __name__ == "__main__":
    train_ds = build_ds("training/*.jpg", batch=32, is_train=True, augment=True)
    val_ds   = build_ds("validation/*.jpg", batch=32, is_train=False)
    test_ds  = build_ds("test/*.jpg", batch=32, is_train=False)

    print("Training dataset (with augmentation):")
    for img, label in train_ds.take(1):
        print(f"Image shape: {img.shape}, Label: {label}")
        print(f"Image range: [{img.numpy().min():.3f}, {img.numpy().max():.3f}]")

    def assert_label_variety(ds, name, max_batches=200):
        import numpy as np
        vals = []
        for i, (_, y) in enumerate(ds.take(max_batches)):
            vals.append(y.numpy().reshape(-1))
        if not vals:
            raise RuntimeError(f"{name}: dataset appears empty.")
        allv = np.concatenate(vals)
        if allv.min() == allv.max():
            raise RuntimeError(
                f"{name}: collapsed labels (min=max={allv.min()})."
            )
        print(f"{name}: count={len(allv)}, min={allv.min()}, max={allv.max()}, mean={allv.mean():.3f}")

    assert_label_variety(val_ds, "Validation")
    assert_label_variety(test_ds, "Test")
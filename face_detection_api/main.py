from face_recognizer import save_face, match_face


def test():
    # save face from reference image of the face
    save_face(single_face_image_path='sample_images/alif-ref1.jpg', face_id=0)
    save_face(single_face_image_path='sample_images/jaber-ref1.jpg', face_id=1)
    save_face(single_face_image_path='sample_images/jaber-ref2.jpg', face_id=1)
    save_face(single_face_image_path='sample_images/jaber-ref3.jpg', face_id=1)

    # images for testing
    testing_image_paths = ['sample_images/alif-ref1.jpg',
                           'sample_images/alif-multiple1.jpg',
                           'sample_images/alif-multiple2.jpg',
                           'sample_images/alif-live-test1.jpg',
                           'sample_images/jaber-test1.jpg',
                           'sample_images/jaber-test2.jpg',
                           'sample_images/jaber-multiple.jpg',
                           'sample_images/jaber-live-test1.jpg',
                           'sample_images/jaber-live-test2.jpg']

    # match images to recognize known faces
    for image_path in testing_image_paths:
        print(f'matches for "{image_path}": {match_face(image_path)}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

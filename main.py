
import os
from hw_7.utils import *
from hw_7.orb import ORB
from hw_7.new_matching import *
from hw_7.localization import *
import pickle
path_img = 'C:\\Users\\adels\PycharmProjects\data\ORB'
import time






def prepare_keypoints(name1, name2):
    orb = ORB(tau=10, harris_tau=512, create_points_brief='normal')
    if not os.path.exists(name1[:name1.index('.')]+'_calc_keypoints_query.pickle'):
        org_img_query = read_image(os.path.join(path_img, name1),True)
        img_query = read_image(os.path.join(path_img, name1))


        current_img = img_query.copy()
        orb.fit(current_img)
        orb.draw_keypoints(org_img_query)
        keypoints_query = orb.keypoints_data
        with open(name1[:name1.index('.')]+'_calc_keypoints_query.pickle', 'wb') as f:
            pickle.dump(keypoints_query, f)
    print('-'*100)
    if not os.path.exists(name2[:name2.index('.')]+'_calc_keypoints_test.pickle'):
        org_img_test = read_image(os.path.join(path_img, name2), True)
        img_test = read_image(os.path.join(path_img, name2))
        current_img = img_test.copy()
        orb.fit(current_img)
        orb.draw_keypoints(org_img_test)
        keypoints_test = orb.keypoints_data
        with open(name2[:name2.index('.')]+'_calc_keypoints_test.pickle', 'wb') as f:
            pickle.dump(keypoints_test, f)



def load_keypoints(name1, name2):

    with open(name1[:name1.index('.')]+'_calc_keypoints_query.pickle', 'rb') as f:
        query = pickle.load(f)

    with open(name2[:name2.index('.')]+'_calc_keypoints_test.pickle', 'rb') as f:
        test = pickle.load(f)

    org_img_query = read_image(os.path.join(path_img, name1), True)


    org_img_test = read_image(os.path.join(path_img, name2), True)


    match = Matcher(query,test)
    # match.cross_check = False
    # match.test_lowe = False
    match.fit()
    matched_keypoints = match.get_matched_keypoints()
    draw_matches(matched_keypoints,org_img_query,org_img_test)
    lcl = Localization(*match.get_matched_keypoints(),type_transform='affine')
    # lcl = Localization(*match.get_matched_keypoints(), type_transform='projective')
    lcl.fit()
    transform_coords = lcl.predict(org_img_query)
    draw_matches(lcl.get_inlier_keypoints(),org_img_query,org_img_test,transform_coords,title='Affine with CrossCheck')




if __name__ == "__main__":
    query_name = 'bears.jpg'
    test_name = 'bears_test.jpg'
    # query_name = 'box.png'
    # test_name = 'box_in_scene.png'
    prepare_keypoints(query_name,test_name)
    load_keypoints(query_name,test_name)


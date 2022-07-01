# Hundai_heavy_industries
현대중공업 공모전

================================================
* Competition 1 모델 성능 평가 검증을 위한 실행 가이드
================================================

1. F:\Competition_No1\test_datasets\test 경로에 있는 0/1 폴더안에 레이블 0과 1인 테스트 데이터를 각각 넣어줍니다.

	ex) test_datasets
                |
                |----test
                         |
                         |--- 0 
                         |      |-- image1.jpg
                         |      |-- image2.jpg
                         |
                         |--- 1
								|-- image3.jpg
                                |-- image4.jpg


2. 바탕화면에 있는 pycharm IDE를 실행시켜서 좌측 상단의 File -> Open을 눌러서 F:\Competition_No1\code 프로젝트를

	열어줍니다. (VM환경의 작업 표시줄에 프로젝트를 열어두었습니다)

3. 좌측에 보이는 test.py 파일을 열어줍니다.

4. 다음 경로가 제대로 되어있는지 확인하고 오른쪽 상단에 있는 화살표 버튼을 눌러 실행합니다. (Ctrl + Enter)

	#########################################################################################
	#  경로 설정
	#########################################################################################
	test_data_set = 'F:/Competition_No1/test_datasets'
	saved_loc = 'F:/Competition_No1/code/saved_model/ckpt.pth' # best 모델 저장 경로
	test_result_path = 'F:/Competition_No1/test_result/test'

5. F:\Competition_No1\test_result\test 경로에 만들어져 있는 테스트 결과를 확인합니다.


================================================
* Competition 2 모델 성능 평가 검증을 위한 실행 가이드
================================================

1. F:\Competition_No2\test_datasets\test 경로에 있는 01/02/03 폴더안에 이미지 데이터와 gt.txt 파일을 각각 넣어줍니다.

	ex) test_datasets
                |
                |----test
                         |
                         |--- 01 
                         |       |-- image1.jpg
                         |       |-- image2.jpg
                         |       |-- gt_test_01.txt
                         |
                         |--- 02
						 |		  |-- image3.jpg
                         |        |-- image4.jpg
                         |        |-- gt_test_02.txt
                         |
                         |--- 03
								 |-- image5.jpg
                                 |-- image6.jpg
                                 |-- gt_test_03.txt


2. 바탕화면에 있는 pycharm IDE를 실행시켜서 좌측 상단의 File -> Open을 눌러서 F:\Competition_No2\code 프로젝트를

	열어줍니다. (VM환경의 작업 표시줄에 프로젝트를 열어두었습니다)

3. 좌측에 보이는 test.py 파일을 열어줍니다.

4. 다음 경로가 제대로 되어있는지 확인하고 오른쪽 상단에 있는 화살표 버튼을 눌러 실행합니다. (Ctrl + Enter)

	- test_data_path : test 이미지가 들어있는 폴더 경로
	- test_label_path : test 레이블 정보가 들어있는 txt파일 경로
	- test_result_path : test 결과 csv 파일이 저장되는 폴더 경로

    #########################################################################################
    #  경로 설정
    #########################################################################################

    # test_data_path = 'F:/Competition_No2/test_datasets/test/01'
    # test_label_path = 'F:/Competition_No2/test_datasets/test/01/gt_test_01.txt'
    # test_result_path = 'F:/Competition_No2/test_result/test/01'
    #
    # test_data_path = 'F:/Competition_No2/test_datasets/test/02'
    # test_label_path = 'F:/Competition_No2/test_datasets/test/02/gt_test_02.txt'
    # test_result_path = 'F:/Competition_No2/test_result/test/02'

    test_data_path = 'F:/Competition_No2/test_datasets/test/03'
    test_label_path = 'F:/Competition_No2/test_datasets/test/03/gt_test_03.txt'
    test_result_path = 'F:/Competition_No2/test_result/test/03'


	* 모델은 세가지 데이터셋 01, 02, 03을 모두 예측할 수 있는 통합 모델이며, 각 데이터셋에 대해서 테스트를 한번씩 수행합니다. 

	이때 테스트를 수행하지 않는 데이터 셋에 대해서는 주석처리 해야하며, 따라서 총 3번 test.py 파일을 실행해야 합니다. 

	(test.py 파일 실행 단축키는 Ctrl + Enter로 설정되어 있으며, data set 01에 대한 테스트를 수행할 경우 02, 03번에 해당하는 코드 주석처리 해야합니다)

5. F:\Competition_No2\test_result\test 경로에 만들어져 있는 테스트 결과를 확인합니다.




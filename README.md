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



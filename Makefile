c_ui: Ui_RubiksRobot.ui
	pyuic5 -o Ui_RubiksRobot.py Ui_RubiksRobot.ui

c_ui2: Ui_RoiWindowDialog.ui
	pyuic5 -o Ui_RoiWindowDialog.py Ui_RoiWindowDialog.ui

r_tmp:
	python tmp1.py
	
r_main:
	python .\ColaMain.py -t run

r_main_test:
	python .\ColaMain.py -t run -is-test

r_show:
	python ColaMain.py -t show

r_sever: server.py
	python server.py

# kaggleSample

kaggleの勉強中

12/12
宮本先輩によるKaggle-lightGBMの勉強会
LightGBM は、2016年に米マイクロソフト社が公開した勾配ブースティングに基づく機械学習手法で、kagglerの6割が使用していると集計結果が出てるほどマストな機械学習手法

宮本先輩から学んだlightGBMは精度が高く、多くの人に使われていることが分かった。
講義中に正規化を実施していない変数があるため、正規化を実施するとより精度があるかもしれないと仰ったので、前処理段階での精度への影響を調べたところ、欠損値を補完しないで、欠損値があるレコードを削除し、lightGBMを使って90％以上の精度を作り出している記事を発見したため、試してみる。
https://banga-heavy.com/%E3%80%90kaggle%E3%80%91タイタニックデータをlightgbmでoptuna%EF%BC%88シンプルに%EF%BC%89/
code:python
 df_train.info() #学習データの中身を確認
 df_test.info() #テストデータの中身を確認
 
 df_train=df_train.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)
 df_test=df_test.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)
 #dropで変数を落としている（消している）
 
 df_train.info() #消去した結果を確認する。
 
 df_train=df_train.dropna()
 df_train.info()
 #年齢も欠損多いので、dropna(欠損値がない行）でないやつ落とし、欠損値がない714個の年齢に合わせる。
 #欠損値を補うのではなく、すべてのデータがあるもので学習を行う方針
 
 x=df_train.drop("Survived",axis=1) #学習データから生存変数の列を削除する
 t=df_train.iloc[:,0:1] #一列目の値（生存変数の列を抽出している）
 #print(x)
 #print(t)
 
 x=df_train.drop("Survived",axis=1) #学習データから生存変数の列を削除する
 t=df_train.iloc[:,0:1] #一列目の値（生存変数の列を抽出している）
データの中身をみて関連性のある変数を残し、さらに欠損値があるレコードを消して、学習データの形成を行う。




12/05
宮本先輩によるKaggle勉強会
宮本先輩から頂いたコードを参考に、説明変数の組み合わせをいくつか試したところ
cols = ['Pclass','Age','Sex','Fare']など、Pclass,Age,Sex以外の要素を増やすと正解率が少し下がってしまった。
説明変数は多ければ多いほど良いというわけでもなく、より細かな条件に当てはまる対象を探すため、かえって正解率が低くなるため、logistic_coefなどを参考に、関連がありそうな説明変数を程よい数決めることが大切だと分かった。

また、欠損値の処理により大きく正解率などに影響してくるため、唯0を入力する以外に、中央値をとなどと工夫が必要。
code:python
 # データの前処理
 train['Sex'] = train['Sex'].map({'male':0, 'female':1})
 test['Sex'] = test['Sex'].map({'male':0, 'female':1})
 
 # 説明変数の選択
 cols = ['Pclass','Age','Sex','Fare']
 
 # 訓練データ
 X_train = train[cols] # 説明変数
 y_train = train['Survived'] # 目的変数
 
 # テストデータ
 X_test = test[cols] # 説明変数
 y_test = test['Survived'] # 目的変数
 
 # モデル作成（ロジスティック回帰）
 model = LogisticRegression(solver='liblinear', random_state=42)
 
 # 学習
 model.fit(X_train, y_train)
 # 予測
 y_pred = model.predict(X_test)
 
 print('正解率：',accuracy_score(y_test, y_pred))

11/21
タイタニックのモデル作成
code:python
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 
 df = pd.read_csv('../input/titanic/train.csv')
 df_test = pd.read_csv('../input/titanic/test.csv')
ここでのcsv読み込みは、KaggleのCodeでdataをinputし、使用したいファイルの上でpathをコピーしたものを貼り付けることで使える。
pathを間違えたら何もできないので注意。

df.describe()を使うことで、平均値や中央値などを確認できる。
今回制作するモデルでは、年齢(age),性別(sex),チケット料金(fare)を使用したかったため、欠損値が無いか確認いたところ、年齢に欠損値が見つかったため，
code:python
 df['Age'].fillna(df['Age'].mean(), inplace=True)
 df_test['Age'].fillna(df_test['Age'].mean(), inplace=True)
年齢の平均値を欠損値に代入をした。
また、性別を数値データとして取り扱うために、
code:python
 df.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
 df_test.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
男性を0,女性を1にし、扱いしやすくした。



11/04
	https://aizine.ai/kaggle-01-0905/ を行った。現在UIが多少変更されているが基本的にそのまま従って出来る。
		初心者向けに入門用のコンペが2つ用意されている。
		もともと用意されているデータを提出してみるチュートリアル。
		アカウント登録をし、タイタニックのコンペページ（https://www.kaggle.com/c/titani ）	にアクセスする。
		Submit PredictionsをクリックするとSMS認証を求められるため登録する。
		登録が完了すると、Notebookを開けるため新しいNotebookを作成する。
		初期状態だとpythonに対応しているため、「print("hello")」と試しに使ってみる。
			実際にはこのNotebookで作成したプログラムを提出する。
		タイタニックのページに戻り、Dataからサンプルファイルをダウンロードして提出をしてみる。
		提出に成功したらランキングに反映され、leaderboardから確認できる。

		今回のタイタニックコンペでは、学習データ（train.csv）には891人の乗客データが与えられ，テストデータ（test.csv）には418人の乗客データが与えられている。
		これら2つのデータを使ってモデルを作成し、解答サンプルに沿った形で出力する。
		完成したファイルを提出すると解答サンプルで実際の精度が測られる。

		もう一つのチュートリアルコンペ、住宅価格予測の中身を詳しく紹介しているサイト（https://note.com/estyle_blog/n/n83fe11a6ca68 )を読む。
		かなり詳しく書かれていて、初歩から紹介されてるため読み込んで理解するのに時間がかかっている。[https://scrapbox.io/files/63720c19099803001ddcaf9f.webp]
		全体の流れが上のようになる。
		効果的な分析ができるように不要なデータを取り除いたりといった前処理が大切

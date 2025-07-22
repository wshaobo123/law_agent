# law_agent
基于 LangGraph 的动态法律分析智能体
1.将.env.example文件复制一份重命名为.env, 将里面的APIKEY改成自己的
2.使用docker运行milvus服务，启动Milvus:  
docker-compose -f milvus-standalone-docker-compose.yml up -d
3.安装依赖:
pip install -r requirements.txt
4.加载数据: (首次运行或数据更新时)
python data_loader.py
5.启动应用:
streamlit run app.py
6.在聊天框进行对话测试

From python:3.8.10

WORKDIR /Home

COPY requirement.txt ./requirement.txt

RUN pip3 install -r requirement.txt

EXPOSE 8502

COPY . /Home

ENTRYPOINT ["streamlit","run"]

CMD ["Home.py"]
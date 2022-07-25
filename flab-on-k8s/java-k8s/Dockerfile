FROM openjdk:8-jre
COPY /target/flab-mng.jar /usr/jar/flab-mng.jar
EXPOSE 18000
CMD java -jar /usr/jar/flab-mng.jar  --spring.profiles.active=test

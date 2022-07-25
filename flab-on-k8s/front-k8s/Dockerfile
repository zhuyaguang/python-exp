FROM nginx:latest
COPY nginx.conf /etc/nginx/nginx.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY mime.types /etc/nginx/mime.types
COPY html/ /etc/nginx/html
COPY cert/ /usr/local/nginx/cert

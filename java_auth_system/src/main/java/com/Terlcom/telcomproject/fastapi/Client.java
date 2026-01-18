package com.Terlcom.telcomproject.fastapi;


import com.Terlcom.telcomproject.dto.ResponseDTO;
import org.springframework.stereotype.Component;
import tools.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

@Component
public class Client {


    public <RequestDto> ResponseDTO previsao(RequestDto requestDto) throws IOException, InterruptedException {
        ObjectMapper mapper = new ObjectMapper();
        String endereco = "Http://localhost:8080/prever";
        String json = mapper.writeValueAsString(requestDto);
        HttpClient client = HttpClient.newHttpClient();

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(endereco))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json))
                .build();


        HttpResponse<String> response = client.send(request,
                HttpResponse.BodyHandlers.ofString());

        return mapper.readValue(response.body(), ResponseDTO.class);
    }

}

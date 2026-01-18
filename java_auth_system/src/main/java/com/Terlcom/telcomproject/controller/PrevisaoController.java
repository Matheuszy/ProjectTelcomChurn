package com.Terlcom.telcomproject.controller;


import com.Terlcom.telcomproject.fastapi.Client;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;


@RestController
@RequestMapping("/prever")
public class PrevisaoController {

    private final Client client;
    @Autowired
    public PrevisaoController(Client client) {
        this.client = client;
    }


    @PostMapping
    public <RequestDto> ResponseEntity<RequestDto> previsao(@RequestBody RequestDto requestDto) {
        return new ResponseEntity<>(requestDto, HttpStatus.OK);

    }

}

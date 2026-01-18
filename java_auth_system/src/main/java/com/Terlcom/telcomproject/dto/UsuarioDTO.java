package com.Terlcom.telcomproject.dto;


import com.Terlcom.telcomproject.model.UsuarioEnum;

public record UsuarioDTO(
        String userName,
        String email,
        String password,
        UsuarioEnum role

) {
}
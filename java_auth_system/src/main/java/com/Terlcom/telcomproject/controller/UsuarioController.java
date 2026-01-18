package com.Terlcom.telcomproject.controller;

import com.Terlcom.telcomproject.dto.ResponseUsuarioDTO;
import com.Terlcom.telcomproject.dto.UsuarioDTO;
import com.Terlcom.telcomproject.model.Usuario;
import com.Terlcom.telcomproject.service.UsuarioService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/usuario")
public class UsuarioController {

    private final UsuarioService usuarioService;
    private final PasswordEncoder passwordEncoder;


    public UsuarioController(UsuarioService usuarioService, PasswordEncoder passwordEncoder) {
        this.usuarioService = usuarioService;
        this.passwordEncoder = passwordEncoder;
    }

    @PostMapping()
    public ResponseEntity<ResponseUsuarioDTO> novoUsuario(
            @RequestBody UsuarioDTO dto
    ) {
        Usuario usuario = new Usuario(
                dto.userName(),
                dto.email(),
                dto.password(),
                dto.role()
        );

        Usuario salvo = usuarioService.createUsuario(usuario);

        return ResponseEntity.ok(
                new ResponseUsuarioDTO(
                        salvo.getId(),
                        salvo.getUsername(),
                        salvo.getEmail()
                )
        );
    }
}


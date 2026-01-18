package com.Terlcom.telcomproject.controller;

import com.Terlcom.telcomproject.Repositores.UsuarioRepository;
import com.Terlcom.telcomproject.dto.LoginRequestDTO;
import com.Terlcom.telcomproject.dto.LoginResponseDTO;
import com.Terlcom.telcomproject.model.Usuario;
import com.Terlcom.telcomproject.service.AuthService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/auth")
public class AuthController {

    private final AuthService authService;
    private final UsuarioRepository usuarioRepository;
    private final AuthenticationManager authenticationManager;

    public AuthController(AuthService authService, UsuarioRepository usuarioRepository, AuthenticationManager authenticationManager) {
        this.authService = authService;
        this.usuarioRepository = usuarioRepository;
        this.authenticationManager = authenticationManager;
    }

    @PostMapping("/login")
    public ResponseEntity<LoginResponseDTO> login(@RequestBody LoginRequestDTO dto) {
        var token = new UsernamePasswordAuthenticationToken(dto.userName(), dto.password());
        var authentication = authenticationManager.authenticate(token);

        Usuario usuario = (Usuario) authentication.getPrincipal();


        return ResponseEntity.ok(new LoginResponseDTO(
                usuario.getUsername(),
                usuario.getRole().name()
        ));
    }
}

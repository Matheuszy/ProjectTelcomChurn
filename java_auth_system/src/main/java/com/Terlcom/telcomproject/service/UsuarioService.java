package com.Terlcom.telcomproject.service;

import com.Terlcom.telcomproject.Repositores.UsuarioRepository;
import com.Terlcom.telcomproject.model.Usuario;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class UsuarioService {

    private final UsuarioRepository usuarioRepository;
    private final PasswordEncoder passwordEncoder;


    public UsuarioService(UsuarioRepository usuarioRepository, PasswordEncoder passwordEncoder) {
        this.usuarioRepository = usuarioRepository;
        this.passwordEncoder = passwordEncoder;
    }


    public Usuario createUsuario(Usuario usuario) {
        if (usuarioRepository.findByUserName(usuario.getUsername()).isPresent()) {
            throw new RuntimeException("Username já existe!");
        } else if (usuarioRepository.findByEmail(usuario.getEmail()).isPresent()) {
            throw new RuntimeException("Email já cadastrado");
        }
        usuario.setPassword(passwordEncoder.encode(usuario.getPassword()));
        return usuarioRepository.save(usuario);
    }


}

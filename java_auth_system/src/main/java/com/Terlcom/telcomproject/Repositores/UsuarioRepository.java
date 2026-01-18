package com.Terlcom.telcomproject.Repositores;

import com.Terlcom.telcomproject.dto.UsuarioDTO;
import com.Terlcom.telcomproject.model.Usuario;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;
import java.util.UUID;
@Repository
public interface UsuarioRepository extends JpaRepository<Usuario, UUID> {

    Optional<Usuario> findByUserName(String userName);

    Optional<Usuario> findByEmail(String email);

}

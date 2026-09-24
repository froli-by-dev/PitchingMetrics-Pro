from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# ==========================================
# 1. CONEXIÓN A LA BASE DE DATOS SQLITE
# ==========================================
DB_URL = "sqlite:///pitching_app.db"
engine = create_engine(DB_URL, echo=False)

Base = declarative_base()

# ==========================================
# 2. TABLAS (MODELOS RELACIONALES)
# ==========================================

class Pitcher(Base):
    __tablename__ = 'pitchers'

    id = Column(Integer, primary_key=True)
    nombre = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)

    analisis = relationship("AnalisisBiomecanico", back_populates="pitcher", cascade="all, delete-orphan")


class AnalisisBiomecanico(Base):
    __tablename__ = 'analisis_biomecanico'

    id = Column(Integer, primary_key=True)
    pitcher_id = Column(Integer, ForeignKey('pitchers.id'), nullable=False)
    fecha = Column(DateTime, default=datetime.now)
    
    angulo_codo = Column(Float, nullable=False)
    angulo_hombro = Column(Float, nullable=False)
    sugerencia = Column(String(500), nullable=False)

    pitcher = relationship("Pitcher", back_populates="analisis")


# Crear las tablas físicamente
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)


# ==========================================
# 3. LÓGICA DE PRUEBA SEGURA (SIN ERRORES)
# ==========================================

def ejecutar_demo():
    session = SessionLocal()
    email_test = "cesar123@ejemplo.com"
    
    # Busca si el pitcher ya existe en la base de datos
    pitcher = session.query(Pitcher).filter(Pitcher.email == email_test).first()

    # Si NO existe, lo crea
    if not pitcher:
        pitcher = Pitcher(
            nombre="César Castillo",
            email=email_test,
            password_hash="contrasena_encriptada_123"
        )
        session.add(pitcher)
        session.commit()
        session.refresh(pitcher)
        print(f" Pitcher '{pitcher.nombre}' registrado con éxito (ID #{pitcher.id}).")
    else:
        print(f"ℹ El pitcher con email '{email_test}' ya existe en la base de datos.")

    # Agrega un análisis a su historial
    nuevo_analisis = AnalisisBiomecanico(
        pitcher_id=pitcher.id,
        fecha=datetime.now(),
        angulo_codo=85.0,
        angulo_hombro=90.0,
        sugerencia="Mantener la extensión del codo y buena rotación de cadera."
    )
    session.add(nuevo_analisis)
    session.commit()
    print(f" Nuevo análisis agregado al historial a las {nuevo_analisis.fecha.strftime('%H:%M:%S')}.")

    # Mostrar historial en pantalla
    print(f"\n================ HISTORIAL DE {pitcher.nombre.upper()} ================")
    for item in pitcher.analisis:
        print(f"Fecha: {item.fecha.strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"  • Codo: {item.angulo_codo}° | Hombro: {item.angulo_hombro}°")
        print(f"  • Nota: {item.sugerencia}")
        print("-" * 55)

    session.close()


if __name__ == "__main__":
    ejecutar_demo() 
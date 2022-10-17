import "./App.css";
import { useEffect, useState } from "react";
import {
  Box,
  Button,
  CircularProgress,
  IconButton,
  Paper,
  Tooltip,
} from "@mui/material";
import {
  AddAPhoto,
  AddAPhotoOutlined,
  AutoAwesome,
  AutoMode,
  EmojiPeople,
  PlayArrowOutlined,
  PlayCircle,
  SendTimeExtension,
  SensorOccupied,
  Settings,
  StopOutlined,
} from "@mui/icons-material";

const video = "http://localhost:5000/stream_video";
const baseUrl = "http://localhost:5000";

function App() {
  const [isVideoLoaded, setIsVideoLoaded] = useState(false);
  const [streamState, setStreamState] = useState({});

  const buttons = [
    {
      name: "start_stream",
      label: "Iniciar Transmissão",
      icon: <PlayArrowOutlined />,
    },
    {
      name: "stop_stream",
      label: "Parar Transmissão",
      icon: <StopOutlined />,
    },
    {
      name: "add_photo",
      label: "Registrar Foto",
      icon: <AddAPhotoOutlined />,
    },
    {
      name: "set_detect",
      label: "Detectar Objetos",
      icon: <SensorOccupied />,
    },
    {
      name: "set_detect_emotion",
      label: "Detectar Emoção",
      icon: <EmojiPeople />,
    },
    {
      name: "set_detect_moviment",
      label: "Detectar Movimento",
      icon: <AutoMode />,
    },
    {
      name: "set_detect_mask",
      label: "Detectar Objetos (Avançado)",
      icon: <SendTimeExtension />,
    },
    {
      name: "set_timelapse",
      label: "Timelapse ",
      icon: <AutoAwesome />,
    },
    {
      name: "configure",
      label: "Configurar ",
      icon: <Settings />,
    },
  ];
  useEffect(() => {
    fetch(`${baseUrl}/get_state`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    }).then((resp) => setStreamState(resp));
  }, []);

  const onLoadedData = () => {
    setIsVideoLoaded(true);
  };
  const Video = () => {
    const handleConfigure = () => {};

    return (
      <div className="container">
        <img src={video} alt="logo" onLoad={onLoadedData} />
        {!isVideoLoaded && <CircularProgress />}
      </div>
    );
  };
  const handleConfigureSubmit = async (body) => {
    await fetch(`${baseUrl}/configure`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    }).then((resp) => console.log(resp));
  };

  return (
    <div className="App">
      <header className="App-header">
        <Video />

        <Box component={Paper} p={1} mt={2}>
          {buttons.map((button) => (
            <Tooltip title={button.label}>
              <IconButton onClick={() => handleConfigureSubmit(button.name)}>
                {button.icon}
              </IconButton>
            </Tooltip>
          ))}
          {/* <IconButton
            onClick={() => handleConfigureSubmit("run_stream")}
            disabled={!isVideoLoaded}
          >
            <PlayCircle />
          </IconButton>
          <IconButton
            onClick={() => handleConfigureSubmit("add_photo")}
            disabled={!isVideoLoaded}
          >
            <AddAPhoto />
          </IconButton>
          <IconButton
            onClick={() => handleConfigureSubmit("set_detect")}
            disabled={!isVideoLoaded}
          >
            <SensorOccupied />
          </IconButton>
           <IconButton>
            <InsertEmoticon />
          </IconButton> 
          <IconButton
            onClick={() => handleConfigureSubmit("set_timelapse")}
            disabled={!isVideoLoaded}
          >
            <AutoMode />
          </IconButton>
          <IconButton
            onClick={() => handleConfigureSubmit("set_detect_emotion")}
            disabled={!isVideoLoaded}
          >
            <EmojiPeople />
          </IconButton>
          <IconButton
          // onClick={() => handleConfigureSubmit("start_stream")}
          // disabled={!isVideoLoaded}
          >
            <Settings />
          </IconButton>{" "}
          */}
        </Box>
      </header>
    </div>
  );
}

export default App;

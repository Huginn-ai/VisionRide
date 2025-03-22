# Version: 2024/12/19

class ClientSession:
    """
    This class is used to store variables for each client, and when each client connects to the server, an instance is created. 
    This enables isolation of temporary data between different clients.
    """
    def __init__(self, websocket):
        self.websocket = websocket

        self.__init_setting()

    def reset(self):
        self.__init_setting()

    def __init_setting(self):
        '''This command is used to initialize all data'''
        self.client_is_playing_audio = False
        

    async def send_message(self, message):
        await self.websocket.send(message)

    async def receive_message(self):
        return await self.websocket.recv()

    async def close(self):
        await self.websocket.close()


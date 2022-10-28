import os
import pickle
import warnings
import numpy as np
from sklearn.cluster import KMeans
from ExtratorCaracteristicas import extract_mfcc, extract_stft
from librosa import load
from scipy import signal
from pydub.silence import split_on_silence
from pydub import AudioSegment 


def silence_eliminator(input_path, output_path):
	aud = AudioSegment.from_wav(input_path)
	
	audio_chunks = split_on_silence(aud, min_silence_len = 100, silence_thresh = -60, keep_silence = 10)
	combined = AudioSegment.empty()

	for chunck in audio_chunks:
		combined += chunck
		
	combined.export(f'{output_path}', format = 'wav')
	signal, sample_rate = load(output_path, sr = 8000)
	no_silence_duration = signal.shape[0] / sample_rate

	os.remove(output_path)
	return signal, no_silence_duration


def training_models(op):
	file_paths = []

	# obtendo o caminho dos arquivos
	for path in os.walk("TrainingData"):
		aux_file_paths = [] # caminho dos arquivos
		for file in path[2]:
			aux_file_paths.append( os.path.join(path[0], file) ) # path[0] = root, path[2] = path
		if aux_file_paths:
			file_paths.append(aux_file_paths)

	# extraindo as features de cada speaker
	for path in file_paths:
		features = np.asarray(())
		for file_path in path:
			print ("Caminho do arquivo:", file_path)
			
			try:
				files_without_silence_path = "temp-" + os.path.basename(file_path).split('.')[0] + ".wav"
				
				# sem remover o silencio ainda
				audio_infos = load(file_path, sr = 8000)
				audio = audio_infos[0]
				audio_rate = audio_infos[1]
				
				# removendo o silencio
				new_audio_infos = silence_eliminator(file_path, files_without_silence_path)
				audio = new_audio_infos[0]
				
				# filtro savgol
				audio = signal.savgol_filter(audio, window_length = 3, polyorder = 2, mode = 'nearest')
				
				# extrator caracteristicas
																				### com pre/process        sem pre/process
				
				if op == "mfcc":
					vector = extract_mfcc(audio, audio_rate)                       ### result - 82.490        result - 95,642
				if op == "stft":
					vector = extract_stft(audio, audio_rate)                       ### result - 64.669        result - 81,167

				#vector = features_extractor.extract_bfcc(audio, audio_rate)       ### result - 55,157
				#vector = features_extractor.extract_gfcc(audio, audio_rate)       ### result - 73,230
				#vector = features_extractor.extract_lfcc(audio, audio_rate)       ### result - 92,918
				#vector = features_extractor.extract_lpcc(audio, audio_rate)       ### result - 06,148
				#vector = features_extractor.extract_msrcc(audio, audio_rate)      ### result - 74,553
				#vector = features_extractor.extract_rplp(audio, audio_rate)       ### result - 13,230   

			except NameError:
				continue

			if features.size:
				features = np.vstack((features, vector))
			else:
				features = vector
				
		# configurando vq
		vq = KMeans(n_clusters = 16, n_init = 3, random_state = None)
		vq.fit(features)

		# guardando os modelos de treino
		training_file = "ModelosLocutores/" + os.path.basename(file_path).split('_')[0] + ".vq"
		with open(training_file, 'wb') as vq_file:
			pickle.dump(vq, vq_file)

		print('\nModelo de locutor:', training_file, 'concluido\n')


def speaker_identifier(op):
	# coletando os caminhos dos arquivos .vq e carregando em db
	db = {}

	for file_name in os.listdir("ModelosLocutores/"):
		speaker = file_name.split('.')[0]
		model = pickle.load(open(os.path.join("ModelosLocutores/", file_name), 'rb'))
		db[speaker] = model

	file_paths = []

	# coletando os paths dos arquivos
	for paths in os.walk("TestingData"):
		for file in paths[2]:
			file_paths.append(os.path.join(paths[0], file))

	error = 0
	total = 0

	# lendo o TestingData e pegando a lista dos audios de teste
	for path in file_paths:
		if os.path.basename(path).split('_')[0] in db.keys():

			signal, sample_rate = load(path, sr = 8000)

			if op == "mfcc":
				vector = extract_mfcc(signal, sample_rate)
			if op == "stft":
				vector = extract_stft(signal, sample_rate)
			
			if vector.shape != (0,):
				total += 1
				log_likelihood = {}
				# m = {}
				for speaker, model in db.items():
					vq = model
					scores = np.array(vq.score(vector))
					log_likelihood[speaker] = round(scores.sum(), 3)
					# m[speaker] = scores

				max_log_likelihood = max(log_likelihood.values())
				for key, value in log_likelihood.items():
					if value == max_log_likelihood:
						predicted_speaker = key

				expected_speaker = os.path.basename(path).split("_")[0]
				if predicted_speaker != expected_speaker:
					error += 1

			print("ÁUDIO:", os.path.basename(path))
			print("LOCUTOR ESPERADO:", os.path.basename(path).split("_")[0])
			print("LOCUTOR IDENTIFICADO:", predicted_speaker)
			print()


	accuracy = ((total - error) / total) * 100
	print("ACURÁCIA:", round(accuracy, 3))


if __name__== "__main__":
	warnings.filterwarnings("ignore")
	
	op = input("[MFCC]\n[STFT]\n>>> ")
	# treinando locutores
	print("TREINANDO...")
	#training_models(op)
	# testando e reconhecendo/identificando locutores
	print("TESTANDO...")
	speaker_identifier(op)